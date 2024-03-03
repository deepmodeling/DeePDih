import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem, ChemicalForceFields
from typing import List


def get_torsions_from_rotamer(mol, rotamer):
    bonds = mol.GetBonds()
    i1, i2 = rotamer
    linked_to_i1 = []
    for bond in bonds:
        if bond.GetBeginAtomIdx() == i1:
            linked_to_i1.append(bond.GetEndAtomIdx())
        elif bond.GetEndAtomIdx() == i1:
            linked_to_i1.append(bond.GetBeginAtomIdx())
    linked_to_i2 = []
    for bond in bonds:
        if bond.GetBeginAtomIdx() == i2:
            linked_to_i2.append(bond.GetEndAtomIdx())
        elif bond.GetEndAtomIdx() == i2:
            linked_to_i2.append(bond.GetBeginAtomIdx())
    torsions = []
    for i in linked_to_i1:
        for j in linked_to_i2:
            if i != i2 and j != i1:
                torsions.append((i, i1, i2, j))
    return torsions


def get_all_torsions(mol):
    all_tors = []
    for tor in get_rotamers(mol):
        all_tors += get_torsions_from_rotamer(mol, tor[1:3])
    return all_tors


def get_rotamers(molecule: Chem.rdchem.Mol):
    """
    Get all possible torsions in a molecule.

    Args:
        molecule: A RDKit molecule object.

    Returns:
        A list of lists, where each inner list contains the indices of the four atoms defining a torsion.
    """
    rotatable_bond = Chem.MolFromSmarts("[!X1:1]!@;-[!X1:2]")
    matches = molecule.GetSubstructMatches(rotatable_bond)
    atomic_nums = [a.GetAtomicNum() for a in molecule.GetAtoms()]
    torsions = []
    for match in matches:
        neighbor1 = [
            atom
            for atom in molecule.GetAtomWithIdx(match[0]).GetNeighbors()
            if atom.GetIdx() != match[1]
        ]
        neighbor1.sort(key=lambda x: x.GetAtomicNum())
        neighbor2 = [
            atom
            for atom in molecule.GetAtomWithIdx(match[1]).GetNeighbors()
            if atom.GetIdx() != match[0]
        ]
        neighbor2.sort(key=lambda x: x.GetAtomicNum())
        if len(neighbor1) > 0 and len(neighbor2) > 0:
            for n1 in neighbor1:
                for n2 in neighbor2:
                    torsions.append(
                        [n1.GetIdx(), match[0], match[1], n2.GetIdx()])

    # find O-C-N-H rotamers
    peptide_rotamers = []
    for tor in torsions:
        if (
            atomic_nums[tor[0]] == 8
            and atomic_nums[tor[1]] == 6
            and atomic_nums[tor[2]] == 7
            and atomic_nums[tor[3]] == 1
        ):
            peptide_rotamers.append((tor[1], tor[2]))
        elif (
            atomic_nums[tor[0]] == 1
            and atomic_nums[tor[1]] == 7
            and atomic_nums[tor[2]] == 6
            and atomic_nums[tor[3]] == 8
        ):
            peptide_rotamers.append((tor[1], tor[2]))

    NH_torsions = []
    ret = []
    for torsion in torsions:
        n_tor, h_tor = False, False
        if atomic_nums[torsion[0]] == 1:
            h_tor = True
            if atomic_nums[torsion[1]] == 7:
                n_tor = True
        elif atomic_nums[torsion[3]] == 1:
            h_tor = True
            if atomic_nums[torsion[2]] == 7:
                n_tor = True
        if not h_tor:
            ret.append(torsion)
        if n_tor:
            NH_torsions.append(torsion)
    torsions = ret

    ret = []
    rots = []
    for torsion in torsions:
        rot = (
            (torsion[1], torsion[2])
            if torsion[1] < torsion[2]
            else (torsion[2], torsion[1])
        )
        if rot in rots:
            continue
        rots.append(rot)
        ret.append(torsion)
    torsions = ret

    # remove O-C-N-H rotamers
    ret = []
    for tor in torsions:
        if (tor[1], tor[2]) not in peptide_rotamers and (
            tor[2],
            tor[1],
        ) not in peptide_rotamers:
            ret.append(tor)
    torsions = ret

    # remove nitrile
    # find nitrile via SMARTS
    nitrile = Chem.MolFromSmarts("[NX1]#[CX2]")
    matches = molecule.GetSubstructMatches(nitrile)
    ret = []
    for tor in torsions:
        found_nitrile = False
        for match in matches:
            if (match[0] in tor[:2] and match[1] in tor[:2]) or (
                match[0] in tor[2:] and match[1] in tor[2:]
            ):
                found_nitrile = True
                break
        if not found_nitrile:
            ret.append(tor)
    torsions = ret

    # remove C-C triple bond
    # find C-C triple bond via SMARTS
    triple_bond = Chem.MolFromSmarts("[CX2]#[CX2]")
    matches = molecule.GetSubstructMatches(triple_bond)
    ret = []
    for tor in torsions:
        found_triple_bond = False
        for match in matches:
            if (match[0] in tor[:2] and match[1] in tor[:2]) or (
                match[0] in tor[2:] and match[1] in tor[2:]
            ):
                found_triple_bond = True
                break
        if not found_triple_bond:
            ret.append(tor)
    torsions = ret

    return torsions


def get_impropers(molecule: Chem.rdchem.Mol):
    pos = molecule.GetConformer().GetPositions()
    atoms = [a for a in molecule.GetAtoms()]
    rings = [list(r) for r in Chem.GetSymmSSSR(molecule)]
    N_impr = Chem.MolFromSmarts("[*:1]~[#7X3:2](~[*:3])~[*:4]")
    matches = molecule.GetSubstructMatches(N_impr)
    impropers = []
    for match in matches:
        match_l = [i for i in match]
        match_l[0], match_l[1] = match_l[1], match_l[0]
        center = atoms[match_l[0]]
        if not center.GetIsAromatic():
            for ii in [1, 2]:
                if atoms[match_l[ii]].GetSymbol() == "H":
                    match_l[ii], match_l[3] = match_l[3], match_l[ii]
            ang = dihedral(
                pos[match_l[0]], pos[match_l[1]], pos[match_l[2]], pos[match_l[3]]
            )
            if abs(ang) < 1e-2:
                impropers.append(
                    [match_l[0], match_l[1], match_l[2], match_l[3]])
    return impropers


def get_ring_rotamers(molecule: Chem.rdchem.Mol):
    atoms = [a for a in molecule.GetAtoms()]
    rings = [list(r) for r in Chem.GetSymmSSSR(molecule)]
    ring_dihedrals = []
    for ring in rings:
        natoms = len(ring)
        if natoms < 6:
            continue
        for na in range(natoms):
            if na == natoms - 3:
                if (
                    atoms[ring[na + 1]].GetDegree()
                    != atoms[ring[na + 1]].GetTotalValence()
                ):
                    continue
                if (
                    atoms[ring[na + 2]].GetDegree()
                    != atoms[ring[na + 2]].GetTotalValence()
                ):
                    continue
                ring_dihedrals.append(
                    [ring[na], ring[na + 1], ring[na + 2], ring[0]])
            elif na == natoms - 2:
                if (
                    atoms[ring[na + 1]].GetDegree()
                    != atoms[ring[na + 1]].GetTotalValence()
                ):
                    continue
                if atoms[ring[0]].GetDegree() != atoms[ring[0]].GetTotalValence():
                    continue
                ring_dihedrals.append(
                    [ring[na], ring[na + 1], ring[0], ring[1]])
            elif na == natoms - 1:
                if atoms[ring[0]].GetDegree() != atoms[ring[0]].GetTotalValence():
                    continue
                if atoms[ring[1]].GetDegree() != atoms[ring[1]].GetTotalValence():
                    continue
                ring_dihedrals.append([ring[na], ring[0], ring[1], ring[2]])
            else:
                if (
                    atoms[ring[na + 1]].GetDegree()
                    != atoms[ring[na + 1]].GetTotalValence()
                ):
                    continue
                if (
                    atoms[ring[na + 2]].GetDegree()
                    != atoms[ring[na + 2]].GetTotalValence()
                ):
                    continue
                ring_dihedrals.append(
                    [ring[na], ring[na + 1], ring[na + 2], ring[na + 3]]
                )
    return ring_dihedrals


def getRingAtoms(rdmol, join=False):
    ring_atoms = []
    for ring in Chem.GetSymmSSSR(rdmol):
        ratoms = list(ring)
        if join:
            for a in ratoms:
                ring_atoms.append(a)
        else:
            ring_atoms.append(ratoms)
    if join:
        ring_atoms = list(set(ring_atoms))
    return sorted(ring_atoms)


def getMergedRings(rdmol):
    merged_rings = []
    rings = [r for r in Chem.GetSymmSSSR(rdmol)]
    is_aromatic_ring = [False for _ in rings]
    for i, ring in enumerate(rings):
        for atom in ring:
            if rdmol.GetAtomWithIdx(atom).GetIsAromatic():
                is_aromatic_ring[i] = True
                break

    merged_ring_aromatic = []
    for nring, ring in enumerate(rings):
        # check if the atoms in ring are already in merged_rings
        merged = False
        for nmerged in range(len(merged_rings)):
            if merged_ring_aromatic[nmerged] and is_aromatic_ring[nring]:
                if len(set(ring) & set(merged_rings[nmerged])) > 0:
                    merged_rings[nmerged] = list(
                        set(ring) | set(merged_rings[nmerged]))
                    merged = True
                    break
        if not merged:
            ratoms = list(ring)
            merged_rings.append(ratoms)
            merged_ring_aromatic.append(is_aromatic_ring[nring])
    return merged_rings


def get_peptide_bonds(molecule: Chem.rdchem.Mol):
    peptide_bond = Chem.MolFromSmarts("[CX3](=O)[NX3]([H])")
    matches = molecule.GetSubstructMatches(peptide_bond)
    rotamers = []
    for match in matches:
        if match[0] < match[2]:
            rotamers.append((match[0], match[2]))
        else:
            rotamers.append((match[2], match[0]))
    return rotamers


def get_torsions(
    molecule: Chem.rdchem.Mol,
    deduplicate: bool = False,
    no_hydrogen: bool = False,
    N_hydrogen: bool = False,
) -> List[List[int]]:
    """
    Get all possible torsions in a molecule.

    Args:
        molecule: A RDKit molecule object.
        deduplicate: A boolean indicating whether to remove duplicate torsions.
        no_hydrogen: A boolean indicating whether to exclude hydrogen atoms.

    Returns:
        A list of lists, where each inner list contains the indices of the four atoms defining a torsion.
    """
    rotatable_bond = Chem.MolFromSmarts("[!X1:1]!@;-[!X1:2]")
    matches = molecule.GetSubstructMatches(rotatable_bond)
    atomic_nums = [a.GetAtomicNum() for a in molecule.GetAtoms()]
    torsions = []
    for match in matches:
        neighbor1 = [
            atom
            for atom in molecule.GetAtomWithIdx(match[0]).GetNeighbors()
            if atom.GetIdx() != match[1]
        ]
        neighbor1.sort(key=lambda x: x.GetAtomicNum())
        neighbor2 = [
            atom
            for atom in molecule.GetAtomWithIdx(match[1]).GetNeighbors()
            if atom.GetIdx() != match[0]
        ]
        neighbor2.sort(key=lambda x: x.GetAtomicNum())
        if len(neighbor1) > 0 and len(neighbor2) > 0:
            for n1 in neighbor1:
                for n2 in neighbor2:
                    torsions.append(
                        [n1.GetIdx(), match[0], match[1], n2.GetIdx()])

    # find O-C-N-H rotamers
    peptide_rotamers = []
    for tor in torsions:
        if (
            atomic_nums[tor[0]] == 8
            and atomic_nums[tor[1]] == 6
            and atomic_nums[tor[2]] == 7
            and atomic_nums[tor[3]] == 1
        ):
            peptide_rotamers.append((tor[1], tor[2]))
        elif (
            atomic_nums[tor[0]] == 1
            and atomic_nums[tor[1]] == 7
            and atomic_nums[tor[2]] == 6
            and atomic_nums[tor[3]] == 8
        ):
            peptide_rotamers.append((tor[1], tor[2]))

    if no_hydrogen:
        NH_torsions = []
        ret = []
        for torsion in torsions:
            n_tor, h_tor = False, False
            if atomic_nums[torsion[0]] == 1:
                h_tor = True
                if atomic_nums[torsion[1]] == 7:
                    n_tor = True
            elif atomic_nums[torsion[3]] == 1:
                h_tor = True
                if atomic_nums[torsion[2]] == 7:
                    n_tor = True
            if not h_tor:
                ret.append(torsion)
            if n_tor:
                NH_torsions.append(torsion)
        torsions = ret

    if deduplicate:
        ret = []
        rots = []
        for torsion in torsions:
            rot = (
                (torsion[1], torsion[2])
                if torsion[1] < torsion[2]
                else (torsion[2], torsion[1])
            )
            if rot in rots:
                continue
            rots.append(rot)
            ret.append(torsion)
        torsions = ret
    if no_hydrogen and N_hydrogen:
        for tor in NH_torsions:
            torsions.append(tor)

    # remove O-C-N-H rotamers
    ret = []
    for tor in torsions:
        if (tor[1], tor[2]) not in peptide_rotamers and (
            tor[2],
            tor[1],
        ) not in peptide_rotamers:
            ret.append(tor)
    torsions = ret

    # remove nitrile
    # find nitrile via SMARTS
    nitrile = Chem.MolFromSmarts("[NX1]#[CX2]")
    matches = molecule.GetSubstructMatches(nitrile)
    ret = []
    for tor in torsions:
        found_nitrile = False
        for match in matches:
            if (match[0] in tor[:2] and match[1] in tor[:2]) or (
                match[0] in tor[2:] and match[1] in tor[2:]
            ):
                found_nitrile = True
                break
        if not found_nitrile:
            ret.append(tor)
    torsions = ret

    # remove C-C triple bond
    # find C-C triple bond via SMARTS
    triple_bond = Chem.MolFromSmarts("[CX2]#[CX2]")
    matches = molecule.GetSubstructMatches(triple_bond)
    ret = []
    for tor in torsions:
        found_triple_bond = False
        for match in matches:
            if (match[0] in tor[:2] and match[1] in tor[:2]) or (
                match[0] in tor[2:] and match[1] in tor[2:]
            ):
                found_triple_bond = True
                break
        if not found_triple_bond:
            ret.append(tor)
    torsions = ret

    return torsions
