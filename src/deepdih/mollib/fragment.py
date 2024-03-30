from typing import Union, List, Dict
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from pathlib import Path
from dpdata.rdkit.sanitize import super_sanitize_mol
import networkx as nx
import numpy as np
import json
from ..utils import (
    DeePDihError,
    getRingAtoms,
    get_rotamers,
    rdmol2graph,
    get_mol_with_indices,
    constrained_embed,
    get_torsions,
    regularize_aromaticity,
    map_mol1_to_mol2,
    fix_aromatic
)
from ..utils import TorEmbeddedMolecule


class Fragmentation(object):
    """
    Class for fragmenting a molecule.

    Args:
        mol: A RDKit molecule object.
        name: A string representing the name of the molecule.

    Attributes:
        mol: A RDKit molecule object.
        name: A string representing the name of the molecule.
        atom_selection: A list of booleans representing whether an atom is selected or not.
        bond_selection: A list of booleans representing whether a bond is selected or not.
        selected_atoms: A list of integers representing the indices of the selected atoms.
        selected_bonds: A list of integers representing the indices of the selected bonds.
        rings: A list of lists, where each inner list contains the indices of the atoms in a ring.
        ring_selection: A list of booleans representing whether a ring is selected or not.
        ortho_pos: A dictionary where the keys are bond indices and the values are lists of bond indices representing ortho positions.

    """

    def __init__(self, mol: Chem.rdchem.Mol, name: str) -> None:
        """
        Initialize the Fragmentation class.

        Args:
            mol: A RDKit molecule object.
            name: A string representing the name of the molecule.

        Returns:
            None
        """
        self.mol = mol
        # Chem.Kekulize(mol, True)
        self.name = name

        self.atom_selection = [False for _ in range(self.mol.GetNumAtoms())]
        self.bond_selection = [False for _ in range(self.mol.GetNumBonds())]
        self.selected_atoms = []
        self.selected_bonds = []

        rings = Chem.GetSymmSSSR(self.mol)
        self.rings = [list(r) for r in rings]
        self.ring_selection = [False for _ in range(len(self.rings))]

        ortho_pos = dict([(bond_idx, [])
                         for bond_idx in range(self.mol.GetNumBonds())])
        for match in self.mol.GetSubstructMatches(
            Chem.MolFromSmarts("[*:1]~[R;!H:2]@[R;!H:3]~[*:4]")
        ):
            bond1_idx = self.mol.GetBondBetweenAtoms(
                match[0], match[1]).GetIdx()
            bond2_idx = self.mol.GetBondBetweenAtoms(
                match[2], match[3]).GetIdx()

            ortho_pos[bond1_idx].append(bond2_idx)
            ortho_pos[bond2_idx].append(bond1_idx)

        self.ortho_pos = ortho_pos

        # self.func_groups = mol.GetSubstructMatches(Chem.MolFromSmarts("[!#6;!#1;!R:1]~[#6X3,#6X2,#6X1,!#6;!#1;!R:2]"))

    def get_fragment_from_rotamer(
        self, atomIdx1, atomIdx2, atomIdx3, atomIdx4, addMethyl=True
    ):
        # prepare
        all_ring_atoms = getRingAtoms(self.mol, join=True)

        rotamers = get_rotamers(self.mol)
        break_bonds = []
        for rot in rotamers:
            if atomIdx2 not in rot[1:3] and atomIdx3 not in rot[1:3]:
                break_bonds.append(rot)
        graph = rdmol2graph(self.mol, bond_to_break=break_bonds)

        subgraph = [
            graph.subgraph(indices) for indices in nx.connected_components(graph)
        ]
        sub_idx = [[n for n in n0.nodes()] for n0 in subgraph]
        for sub in sub_idx:
            if atomIdx2 in sub and atomIdx3 in sub:
                break
        for aidx in sub:
            self.add_atom(aidx)

        # add rings
        self.get_selected_atoms()
        for ringId, ring in enumerate(self.rings):
            if self.ring_selection[ringId]:
                continue
            else:
                for atomId in ring:
                    if atomId in self.selected_atoms:
                        self.add_ring_atoms(ring)
                        break

        # add neighbor atoms except C, H
        self.get_selected_atoms()
        for atomIdx in self.selected_atoms:
            atom = self.mol.GetAtomWithIdx(atomIdx)
            for nei in atom.GetNeighbors():
                if (
                    nei.GetAtomicNum() not in [1, 6]
                    and nei.GetIdx() not in all_ring_atoms
                ):
                    self.add_atom(nei.GetIdx())

        # Find C=O connected to self.selected_atoms and add them using self.add_atom
        self.get_selected_atoms()
        for atom_idx, selected in enumerate(self.atom_selection):
            if not selected:
                continue
            atom = self.mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                neighbor_anum = neighbor.GetAtomicNum()
                if neighbor_anum == 6 and neighbor_idx not in all_ring_atoms:
                    # check if there is a double bond oxygen linked to this carbon
                    for nei in neighbor.GetNeighbors():
                        # check if it's oxygen and only link with one atom
                        if nei.GetAtomicNum() == 8 and nei.GetDegree() == 1:
                            self.add_atom(neighbor_idx)
                            self.add_atom(nei.GetIdx())
                            break

        # Find hydrogen atoms connected to self.selected_atoms and add them using self.add_atom
        self.get_selected_atoms()
        for atom_idx, selected in enumerate(self.atom_selection):
            if not selected:
                continue
            atom = self.mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                neighbor_anum = neighbor.GetAtomicNum()
                if neighbor_anum in [1, 8] and neighbor_idx not in all_ring_atoms:
                    self.add_atom(neighbor_idx)

        # update selected atoms
        self.get_selected_atoms()
        print(f"Selected Atoms : {self.selected_atoms}")

        # if all atoms are selected, return the original molecule
        print(f"Num atoms : {self.mol.GetNumAtoms()}")
        print(f"Num selected atoms : {len(self.selected_atoms)}")
        if len(self.selected_atoms) == self.mol.GetNumAtoms():
            self.clear()
            return self.mol

        # get fragmented bonds
        fragBonds = []
        for atomId in self.selected_atoms:
            atom = self.mol.GetAtomWithIdx(atomId)
            for nei in atom.GetNeighbors():
                if not self.is_atom_selected(nei.GetIdx()):
                    bondId = self.mol.GetBondBetweenAtoms(
                        atomId, nei.GetIdx()).GetIdx()
                    fragBonds.append(bondId)
        assert len(fragBonds) > 0, "No fragment bonds"

        # fragmentation
        frags_ = Chem.FragmentOnBonds(
            self.mol,
            fragBonds,
            bondTypes=[
                Chem.rdchem.BondType.SINGLE for _ in range(len(fragBonds))],
        )
        fragAtoms = []
        frags = Chem.GetMolFrags(
            frags_, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=fragAtoms
        )

        fragId = None
        newAtomsIdx = []
        for i, frag in enumerate(fragAtoms):
            frag_sort = sorted(list(frag))

            # remove dummy indices
            while True:
                if frag_sort[-1] >= self.mol.GetNumAtoms():
                    frag_sort.pop()
                else:
                    break

            # get corrected fragment
            if frag_sort == sorted(self.selected_atoms):
                for atomIdx in [atomIdx1, atomIdx2, atomIdx3, atomIdx4]:
                    newAtomsIdx.append(list(frag).index(atomIdx))
                fragId = i
                break

        assert fragId is not None, "fragId is None"
        newMol = Chem.RWMol(frags[fragId])

        for newAtom in newMol.GetAtoms():
            newAtom.SetProp("_Cap", "FALSE")
            if newAtom.GetSymbol() == "*":
                nei = newAtom.GetNeighbors()[0]
                if addMethyl:
                    if nei.GetSymbol() in ["H"]:
                        cap = Chem.Atom(1)
                    else:
                        cap = Chem.Atom(6)
                else:
                    cap = Chem.Atom(1)
                cap.SetProp("_Cap", "TRUE")
                newMol.ReplaceAtom(
                    newAtom.GetIdx(), cap, updateLabel=True, preserveProps=False
                )

        newMol = newMol.GetMol()
        newMol.UpdatePropertyCache()
        newMol = Chem.AddHs(newMol)
        for atom in newMol.GetAtoms():
            try:
                cap = atom.GetProp("_Cap")
            except:
                atom.SetProp("_Cap", "TRUE")
        print(">>> sanitize")
        Chem.SanitizeMol(newMol)
        print(">>> sanitize done")
        AllChem.EmbedMolecule(newMol, randomSeed=10)

        newMol_noH = Chem.RemoveHs(newMol)
        ca_smi = Chem.MolToSmiles(newMol_noH)
        inchi_key = Chem.MolToInchiKey(newMol_noH)

        newMol.SetProp("TORSION", "-".join([str(x) for x in newAtomsIdx]))
        newMol.SetProp("SMILES", ca_smi)
        newMol.SetProp("INCHI_KEY", inchi_key)

        self.clear()
        return newMol

    def get_fragment_from_torsion(
        self, atomIdx1, atomIdx2, atomIdx3, atomIdx4, addMethyl=True
    ):
        # prepare
        all_ring_atoms = getRingAtoms(self.mol, join=True)

        # save torsion atom indices
        layer_1, layer_2, layer_3 = [], [], []

        # add rotamer
        self.add_atom(atomIdx2)
        self.add_atom(atomIdx3)
        layer_1.append(atomIdx2)
        layer_1.append(atomIdx3)
        layer_2.append(atomIdx2)
        layer_2.append(atomIdx3)
        layer_3.append(atomIdx2)
        layer_3.append(atomIdx3)
        self.get_selected_atoms()
        print(f"Selected Atoms after adding rotamer: {self.selected_atoms}")

        # add atoms linked to rotamer
        for atom_idx in self.selected_atoms:
            atom = self.mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                self.add_atom(neighbor_idx)
                layer_2.append(neighbor_idx)
                layer_3.append(neighbor_idx)
        self.get_selected_atoms()
        print(f"Selected Atoms after adding 1-4 atoms: {self.selected_atoms}")

        # add 1-5 atoms
        for atom_idx in self.selected_atoms:
            atom = self.mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                self.add_atom(neighbor_idx)
                layer_3.append(neighbor_idx)
        self.get_selected_atoms()
        print(f"Selected Atoms after adding 1-5 atoms: {self.selected_atoms}")

        # add rings if more than 1 atoms included in layer_3
        rings_added = []
        for nring, ring in enumerate(self.rings):
            if len(set(ring) & set(layer_3)) > 1 and len(set(ring) & set(layer_2)) >= 1:
                self.add_ring_atoms(ring)
                rings_added.append(nring)
        self.get_selected_atoms()
        print(f"Selected Atoms after adding rings: {self.selected_atoms}")

        # add C=O linked to rings selected
        for atom_idx in self.selected_atoms:
            atom = self.mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() == 6 and atom_idx in all_ring_atoms:
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    neighbor_anum = neighbor.GetAtomicNum()
                    if neighbor_anum == 8 and neighbor_idx not in all_ring_atoms and neighbor.GetDegree() == 1:
                        self.add_atom(neighbor_idx)
        self.get_selected_atoms()
        print(f"Selected Atoms after adding C_ring=O: {self.selected_atoms}")

        # Find C=O connected to self.selected_atoms and add them using self.add_atom
        for atom_idx in self.selected_atoms:
            atom = self.mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                neighbor_anum = neighbor.GetAtomicNum()
                if neighbor_anum == 6 and neighbor_idx not in all_ring_atoms:
                    # check if there is a double bond oxygen linked to this carbon
                    for nei in neighbor.GetNeighbors():
                        # check if it's oxygen and only link with one atom
                        if nei.GetAtomicNum() == 8 and nei.GetDegree() == 1:
                            self.add_atom(neighbor_idx)
                            self.add_atom(nei.GetIdx())
                            break
        self.get_selected_atoms()
        print(f"Selected Atoms after adding -C=O: {self.selected_atoms}")

        # add C(-O)=O linked to self.selected_atoms if the C is already in the selected atoms
        for atom_idx in self.selected_atoms:
            atom = self.mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() == 6:
                no_linked_to_oxygen = 0
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    neighbor_anum = neighbor.GetAtomicNum()
                    if neighbor_anum == 8:
                        no_linked_to_oxygen += 1
                if no_linked_to_oxygen == 2:
                    for neighbor in atom.GetNeighbors():
                        self.add_atom(neighbor.GetIdx())
        self.get_selected_atoms()
        print(f"Selected Atoms after adding C(-O)=O: {self.selected_atoms}")

        # Find non-C/H atoms linked to layer_3
        for atom_idx in layer_3:
            atom = self.mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                neighbor_anum = neighbor.GetAtomicNum()
                if neighbor_anum not in [1, 6] and neighbor_idx not in all_ring_atoms:
                    self.add_atom(neighbor_idx)
        self.get_selected_atoms()
        print(f"Selected Atoms after adding non-C/H: {self.selected_atoms}")

        # [start] >>> Functional Groups <<<

        # Find P and S atoms in self.selected_atoms linked to O and then add all atoms linked to that S
        for atom_idx in self.selected_atoms:
            atom = self.mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() in [15, 16]:
                no_linked_to_oxygen = 0
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    neighbor_anum = neighbor.GetAtomicNum()
                    if neighbor_anum == 8:
                        no_linked_to_oxygen += 1
                if no_linked_to_oxygen > 1:
                    for neighbor in atom.GetNeighbors():
                        self.add_atom(neighbor.GetIdx())
        self.get_selected_atoms()
        print(f"Selected Atoms after adding S=O: {self.selected_atoms}")

        # Find N(=O)=O atoms in self.selected_atoms and add all atoms linked to that N
        for atom_idx in self.selected_atoms:
            atom = self.mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() == 7:
                no_linked_to_oxygen = 0
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    neighbor_anum = neighbor.GetAtomicNum()
                    if neighbor_anum == 8:
                        no_linked_to_oxygen += 1
                if no_linked_to_oxygen > 1:
                    for neighbor in atom.GetNeighbors():
                        self.add_atom(neighbor.GetIdx())
        self.get_selected_atoms()
        print(f"Selected Atoms after adding N(=O)=O: {self.selected_atoms}")

        # [end] >>> Functional Groups <<<

        # Find hydrogen atoms connected to self.selected_atoms and add them using self.add_atom
        self.get_selected_atoms()
        for atom_idx in self.selected_atoms:
            atom = self.mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                neighbor_anum = neighbor.GetAtomicNum()
                if neighbor_anum in [1] and neighbor_idx not in all_ring_atoms:
                    self.add_atom(neighbor_idx)
        self.get_selected_atoms()
        print(f"Selected Atoms after adding Hs: {self.selected_atoms}")

        # Find carbon linked to aromatic N
        for atom_idx in self.selected_atoms:
            atom = self.mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() == 7 and atom.GetIsAromatic():
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    neighbor_anum = neighbor.GetAtomicNum()
                    if neighbor_anum in [1, 6] and not neighbor.GetIsAromatic():
                        self.add_atom(neighbor_idx)

        # update selected atoms
        self.get_selected_atoms()
        print(f"Selected Atoms : {self.selected_atoms}")

        # if all atoms are selected, return the original molecule
        print(f"Num atoms : {self.mol.GetNumAtoms()}")
        print(f"Num selected atoms : {len(self.selected_atoms)}")
        if len(self.selected_atoms) == self.mol.GetNumAtoms():
            self.clear()
            return self.mol

        # get fragmented bonds
        fragBonds = []
        for atomId in self.selected_atoms:
            atom = self.mol.GetAtomWithIdx(atomId)
            for nei in atom.GetNeighbors():
                if not self.is_atom_selected(nei.GetIdx()):
                    bondId = self.mol.GetBondBetweenAtoms(
                        atomId, nei.GetIdx()).GetIdx()
                    fragBonds.append(bondId)
        assert len(fragBonds) > 0, "No fragment bonds"

        # fragmentation
        frags_ = Chem.FragmentOnBonds(
            self.mol,
            fragBonds,
            bondTypes=[
                Chem.rdchem.BondType.SINGLE for _ in range(len(fragBonds))],
        )
        fragAtoms = []
        frags = Chem.GetMolFrags(
            frags_, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=fragAtoms
        )

        fragId = None
        newAtomsIdx = []
        for i, frag in enumerate(fragAtoms):
            frag_sort = sorted(list(frag))

            # remove dummy indices
            while True:
                if frag_sort[-1] >= self.mol.GetNumAtoms():
                    frag_sort.pop()
                else:
                    break

            # get corrected fragment
            if frag_sort == sorted(self.selected_atoms):
                for atomIdx in [atomIdx1, atomIdx2, atomIdx3, atomIdx4]:
                    newAtomsIdx.append(list(frag).index(atomIdx))
                fragId = i
                break

        assert fragId is not None, "fragId is None"
        newMol = Chem.RWMol(frags[fragId])

        # loop atoms, found * atoms and the atoms connected to it
        bonds = []
        natoms = newMol.GetNumAtoms()
        for i in range(natoms):
            atom = newMol.GetAtomWithIdx(i)
            if atom.GetSymbol() == "*":
                nei = atom.GetNeighbors()[0]
                bonds.append([i, nei.GetIdx()])
        print(bonds)

        idx_to_delete = []
        for cap_aidx, parent_aidx in bonds:
            newAtom = newMol.GetAtomWithIdx(cap_aidx)
            parent = newMol.GetAtomWithIdx(parent_aidx)
            newAtom.SetProp("_Cap", "FALSE")
            if parent.GetSymbol() in ["N"] and parent.GetIsAromatic():
                # newMol.RemoveAtom(cap_aidx)
                idx_to_delete.append(cap_aidx)
            if addMethyl:
                if parent.GetSymbol() in ["H", "C"] or parent.GetIsAromatic():
                    cap = Chem.Atom(1)
                else:
                    cap = Chem.Atom(6)
            else:
                cap = Chem.Atom(1)
            cap.SetProp("_Cap", "TRUE")
            newMol.ReplaceAtom(
                newAtom.GetIdx(), cap, updateLabel=True, preserveProps=False
            )
        for idx in sorted(idx_to_delete, reverse=True):
            newMol.RemoveAtom(idx)

        newMol = newMol.GetMol()
        newMol.UpdatePropertyCache()
        newMol = Chem.AddHs(newMol)
        for atom in newMol.GetAtoms():
            try:
                cap = atom.GetProp("_Cap")
            except:
                atom.SetProp("_Cap", "TRUE")
        # Chem.SanitizeMol(newMol)
        fix_aromatic(newMol)
        try:
            print(">>> try to regularize aromaticity")
            regularize_aromaticity(newMol)
        except Exception as e:
            print(f"regularize aromaticity failed: {e}. Use super sanitize.")
            newMol = super_sanitize_mol(newMol)
        AllChem.EmbedMolecule(newMol, randomSeed=10)

        newMol_noH = Chem.RemoveHs(newMol)
        ca_smi = Chem.MolToSmiles(newMol_noH)
        inchi_key = Chem.MolToInchiKey(newMol_noH)

        newMol.SetProp("TORSION", "-".join([str(x) for x in newAtomsIdx]))
        newMol.SetProp("SMILES", ca_smi)
        newMol.SetProp("INCHI_KEY", inchi_key)

        self.clear()
        return newMol

    def is_atom_selected(self, atomId):
        return self.atom_selection[atomId]

    def add_ring_atoms(self, ringAtoms):
        nr = len(ringAtoms)
        for i in range(nr):
            self.add_atom(ringAtoms[i])

    def add_atom(self, atomId):
        self.atom_selection[atomId] = True

    def get_selected_atoms(self):
        self.selected_atoms = [
            i for i in range(self.mol.GetNumAtoms()) if self.atom_selection[i]
        ]
        return self.selected_atoms

    def clear(self):
        self.atom_selection = [False for _ in range(self.mol.GetNumAtoms())]
        self.bond_selection = [False for _ in range(self.mol.GetNumBonds())]
        self.ring_selection = [False for _ in range(len(self.rings))]
        self.selected_atoms = []
        self.selected_bonds = []

    def get_all_fragments(
        self,
        save_sdf: Union[str, Path, None] = None,
        save_png: Union[str, Path, None] = None,
    ):
        torsions = get_torsions(
            self.mol, deduplicate=True, no_hydrogen=True, N_hydrogen=False
        )
        print(f"Find {len(torsions)} torsions.")

        if len(torsions) == 0:
            return []
        elif len(torsions) == 1:
            return [self.mol]

        frags_dict = {}
        n_frag = 0
        for torsion in torsions:
            print(f"Fragmenting on torsion {torsion}")
            try:
                frag = self.get_fragment_from_torsion(*torsion, addMethyl=True)
                frag_ntor = len(get_torsions(frag))
                if frag_ntor == 0:
                    continue
                smi = Chem.MolToSmiles(frag, isomericSmiles=False)
                frags_dict[smi] = frag
                n_frag += 1
            except Exception as e:
                print(f"Fragmentation on torsion {torsion} failed", e)
                raise e

        frags = list(frags_dict.values())
        print(
            f"Successfully get {n_frag} fragments, {len(torsions) - n_frag} failed.")

        if save_sdf is not None:
            writer = Chem.SDWriter(str(save_sdf))
            for frag in frags:
                writer.write(frag)
            writer.close()
            print(f"Fragments save to {save_sdf}")

        if save_png is not None:
            hit_bonds = [[]]
            hit_atoms = [[]]
            for frag in frags:
                torsion = list(map(int, frag.GetProp("TORSION").split("-")))
                hit_atoms.append(torsion)
                hit_bonds.append(
                    [
                        frag.GetBondBetweenAtoms(
                            torsion[0], torsion[1]).GetIdx(),
                        frag.GetBondBetweenAtoms(
                            torsion[1], torsion[2]).GetIdx(),
                        frag.GetBondBetweenAtoms(
                            torsion[2], torsion[3]).GetIdx(),
                    ]
                )
            img = Draw.MolsToGridImage(
                [self.mol] + frags,
                molsPerRow=4,
                subImgSize=(300, 300),
                highlightAtomLists=hit_atoms,
                highlightBondLists=hit_bonds,
            )
            img.save(str(save_png))
            print(f"Fragments plot to {save_png}")
        return frags
