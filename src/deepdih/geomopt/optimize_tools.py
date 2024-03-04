from rdkit import Chem
import numpy as np
import networkx as nx
import geometric
import geometric.molecule
from geometric.internal import Dihedral, PrimitiveInternalCoordinates
from typing import List, Set, Dict, Tuple, Union
import tempfile
from pathlib import Path
from ase.calculators.calculator import Calculator
from copy import deepcopy
from ..utils.transform import rdmol2mol, rdmol2graph
from ..utils.geometry import dihedral
from ..utils.topology import get_rotamers
from ..utils.tools import regularize_aromaticity


def enumerate_propers(rdmol: Chem.rdchem.Mol) -> List[Tuple[int, int, int, int]]:
    dihedrals = []
    for bond in rdmol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            if atom1.GetIdx() > atom2.GetIdx():
                atom1, atom2 = atom2, atom1
            for atom in atom1.GetNeighbors():
                if atom.GetIdx() != atom2.GetIdx():
                    atom3 = atom
            for atom in atom2.GetNeighbors():
                if atom.GetIdx() != atom1.GetIdx():
                    atom4 = atom
            dihedrals.append((atom3.GetIdx(), atom1.GetIdx(),
                             atom2.GetIdx(), atom4.GetIdx()))
    return dihedrals


def enumerate_impropers(rdmol: Chem.rdchem.Mol) -> List[Tuple[int, int, int, int]]:
    impropers_list = []
    impropers = {}
    for parser in [
        "[*:1]~[#6X3:2](~[*:3])~[*:4]",
        "[*:1]~[#6X3:2](~[#8X1:3])~[#8:4]",
        "[*:1]~[#7X3$(*~[#15,#16](!-[*])):2](~[*:3])~[*:4]",
        "[*:1]~[#7X3$(*~[#6X3]):2](~[*:3])~[*:4]",
        "[*:1]~[#7X3$(*~[#7X2]):2](~[*:3])~[*:4]",
        "[*:1]~[#7X3$(*@1-[*]=,:[*][*]=,:[*]@1):2](~[*:3])~[*:4]",
        "[*:1]~[#6X3:2](=[#7X2,#7X3+1:3])~[#7:4]"
    ]:
        for match in rdmol.GetSubstructMatches(Chem.MolFromSmarts(parser)):
            i1, center, i2, i3 = match
            if center not in impropers:
                impropers[center] = []
            atom_set = {i1, i2, i3}
            if atom_set not in impropers[center]:
                impropers[center].append(atom_set)
        for center in impropers:
            for aset in impropers[center]:
                aset_list = list(aset)
                impropers_list.append(
                    (aset_list[0], center, aset_list[1], aset_list[2]))
    return impropers_list


def get_rotamers_from_graph(rdmol: Chem.rdchem.Mol) -> List[Tuple[int]]:
    hydrogen_indices = [i.GetIdx()
                        for i in rdmol.GetAtoms() if i.GetAtomicNum() == 1]
    # find C-C triple bond
    triple_list = []
    for bond in rdmol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            idx1 = atom1.GetIdx()
            idx2 = atom2.GetIdx()
            triple_list.append(idx1)
            triple_list.append(idx2)
    # find rotamers
    rotamers = []
    graph = rdmol2graph(rdmol)
    # for bond in graph:
    for bond in graph.edges():
        # check if the bond is a triple bond
        if bond[0] in triple_list or bond[1] in triple_list:
            continue
        grpah_cpy = graph.copy()
        # break bond
        grpah_cpy.remove_edge(*bond)
        connect_components = list(nx.connected_components(grpah_cpy))
        # check if we have two isolated fragments
        # check if both two fragments have more than 1 atom
        if nx.number_connected_components(grpah_cpy) == 2:
            no_h_atoms_0 = [i for i in connect_components[0]
                            if i not in hydrogen_indices]
            no_h_atoms_1 = [i for i in connect_components[1]
                            if i not in hydrogen_indices]
            if len(no_h_atoms_0) > 1 and len(no_h_atoms_1) > 1:
                rotamers.append(bond)
    return rotamers


def find_constraint_elements(rdmol: Chem.rdchem.Mol, return_all: bool = False, add_improper: bool = False) -> List[Tuple[int]]:
    rotamers = get_rotamers_from_graph(rdmol)
    constraint_elements = []
    for rotamer in rotamers:
        # find the dihedrals on the rotamer
        dihedrals = []
        ii, jj = rotamer
        bonded_to_ii, bonded_to_jj = list(rdmol.GetAtomWithIdx(ii).GetNeighbors()), list(
            rdmol.GetAtomWithIdx(jj).GetNeighbors())
        for kk in bonded_to_ii:
            for ll in bonded_to_jj:
                if kk.GetIdx() != jj and ll.GetIdx() != ii and kk.GetIdx() != ll.GetIdx():
                    dihedrals.append((kk.GetIdx(), ii, jj, ll.GetIdx()))
        if not return_all:
            # pick the heaviest dihedral
            dihedral_weights = []
            for dihedral in dihedrals:
                dihedral_weights.append(
                    sum([rdmol.GetAtomWithIdx(i).GetAtomicNum() for i in dihedral]))
            constraint_elements.append(dihedrals[np.argmax(dihedral_weights)])
        else:
            constraint_elements.extend(dihedrals)
    if add_improper:
        constraint_elements.extend(enumerate_impropers(rdmol))
    return constraint_elements


def compute_constraint_values(
    rdmol: Chem.rdchem.Mol,
    constraints: List[Tuple[int]]
) -> List[float]:
    pos = rdmol.GetConformer().GetPositions()
    const_vals = []
    for const in constraints:
        if len(const) == 2:
            # compute bond
            ii, jj = const
            const_vals.append(np.linalg.norm(pos[ii] - pos[jj]))
        elif len(const) == 3:
            # compute angle
            ii, jj, kk = const
            v1 = pos[ii] - pos[jj]
            v2 = pos[kk] - pos[jj]
            const_vals.append(
                np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        elif len(const) == 4:
            # compute dihedral
            ii, jj, kk, ll = const
            const_vals.append(dihedral(pos[ii], pos[jj], pos[kk], pos[ll]))
        else:
            raise ValueError(f"Invalid constraint: {const}")
    return const_vals


def rotate_position(rdmol, torsion, angle):
    new_rdmol = deepcopy(rdmol)
    # rotate rdmol
    atom1 = new_rdmol.GetAtomWithIdx(torsion[0])
    atom2 = new_rdmol.GetAtomWithIdx(torsion[1])
    atom3 = new_rdmol.GetAtomWithIdx(torsion[2])
    atom4 = new_rdmol.GetAtomWithIdx(torsion[3])
    current_dihedral = Chem.rdMolTransforms.GetDihedralDeg(
        new_rdmol.GetConformer(),
        atom1.GetIdx(),
        atom2.GetIdx(),
        atom3.GetIdx(),
        atom4.GetIdx(),
    )
    new_dihedral = current_dihedral + angle
    if new_dihedral > 180.0:
        new_dihedral -= 360.0

    Chem.rdMolTransforms.SetDihedralDeg(
        new_rdmol.GetConformer(),
        torsion[0],
        torsion[1],
        torsion[2],
        torsion[3],
        new_dihedral,
    )

    return new_rdmol


def writeFreezeString(freeze) -> str:
    strings = []
    strings.append("$freeze")
    for ii, jj, kk, ll in freeze:
        strings.append(f"dihedral {ii+1} {jj+1} {kk+1} {ll+1}")
    return "\n".join(strings)


def writeConstraintString(cons) -> str:
    strings = []
    strings.append("$set")
    for ii, jj, kk, ll, val in cons:
        if val > 180.0:
            val = val - 360.0
        strings.append(f"dihedral {ii+1} {jj+1} {kk+1} {ll+1} {val:.2f}")
    return "\n".join(strings)


def writeScanString(scans) -> str:
    strings = []
    strings.append("$scan")
    for ii, jj, kk, ll, start, end, steps in scans:
        strings.append(
            f"dihedral {ii+1} {jj+1} {kk+1} {ll+1} {start:.2f} {end:.2f} {steps}"
        )
    return "\n".join(strings)


def optimize_molecule(
    mol: geometric.molecule.Molecule,
    calculator: Calculator,
    charge: int = 0,
    freeze: List[Tuple[int, int, int, int]] = [],
    constraints: List[Tuple[int, int, int, int, float]] = [],
    scans: List[Tuple[int, int, int, int, float, float, int]] = [],
    convergence: str = "NORMAL",
) -> geometric.molecule.Molecule:
    engine = geometric.ase_engine.EngineASE(mol, calculator)
    initial_charges = np.zeros(len(engine.ase_atoms))
    initial_charges[0] = charge
    engine.ase_atoms.set_initial_charges(initial_charges)
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        # tmpdir = Path(".")
        cons_file = tmpdir / "cons.txt"
        cons_file.touch()
        with open(cons_file, "w") as f:
            if freeze is not None and len(freeze) > 0:
                f.write(writeFreezeString(freeze))
            if constraints is not None and len(constraints) > 0:
                f.write(writeConstraintString(constraints))
            if scans is not None and len(scans) > 0:
                f.write(writeScanString(scans))
        prefix = tmpdir / "cons_opt"
        m = geometric.optimize.run_optimizer(
            customengine=engine,
            check=1,
            prefix=str(prefix.absolute()),
            constraints=str(cons_file.absolute()),
            convergence_set="GAU_LOOSE",
            convergence_energy=0.01 / 627.5 if convergence == "NORMAL" else 0.1 / 627.5,
            hessian="never",
            enforce=3.0 / 180.0 * np.pi,
            maxiter=128 if convergence == "NORMAL" else 256,
        )
    # import os
    # files = [i for i in os.listdir(".") if "cons_opt_" in i and ".log" in i]
    # os.system(f"cp cons_opt_optim.xyz cons_{len(files)}.xyz")
    return m
