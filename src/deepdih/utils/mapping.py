import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem, ChemicalForceFields


def map_mol1_to_mol2(
    mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol, map_external: bool = True
):
    """
    Map the atom indices of mol1 to mol2.

    Args:
        mol1: A RDKit molecule object.
        mol2: A RDKit molecule object.

    Returns:
        A dictionary where the keys are the atom indices in mol1 and the values are the corresponding atom indices in mol2.
    """
    print(">>> start mapping")
    mcs = rdFMCS.FindMCS(
        [mol1, mol2],
        verbose=True,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=60,
        # atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
    )
    p = Chem.MolFromSmarts(mcs.smartsString)
    m1 = mol1.GetSubstructMatch(p)
    m2 = mol2.GetSubstructMatch(p)

    mapping = dict(zip(m1, m2))
    print(">>> mapping finished")

    if not map_external:
        return mapping

    unmapped_m1 = set(range(mol1.GetNumAtoms())) - set(m1)
    unmapped_m2 = set(range(mol2.GetNumAtoms())) - set(m2)

    m1_atoms = [a for a in mol1.GetAtoms()]
    m2_atoms = [a for a in mol2.GetAtoms()]
    external_bond_m1 = {}
    external_bond_m2 = {}

    for i in unmapped_m1:
        for j in m1:
            if mol1.GetBondBetweenAtoms(i, j) is not None:
                if j not in external_bond_m1:
                    external_bond_m1[j] = []
                external_bond_m1[j].append(i)
                break
    for i in unmapped_m2:
        for j in m2:
            if mol2.GetBondBetweenAtoms(i, j) is not None:
                if j not in external_bond_m2:
                    external_bond_m2[j] = []
                external_bond_m2[j].append(i)
                break

    # if an atom in mol2 is methyl C and has unmapped H, remove the other Hs from mapping dict
    for i in unmapped_m2:
        if m2_atoms[i].GetAtomicNum() == 6 and m2_atoms[i].GetTotalNumHs() == 3:
            # pick up the hydrogens linked with m2_atoms[i]
            hydrogens_to_a2 = []
            for j in m2:
                if mol2.GetBondBetweenAtoms(i, j) is not None:
                    if m2_atoms[j].GetAtomicNum() == 1:
                        hydrogens_to_a2.append(j)
            # check if all hydrogens in hydrogens_to_a2 are mapped
            all_hydrogens_mapped = True
            for j in hydrogens_to_a2:
                if j not in mapping.values():
                    all_hydrogens_mapped = False
                    break
            # if not all hydrogens are mapped, remove the other hydrogens from mapping dict
            if not all_hydrogens_mapped:
                for j in hydrogens_to_a2:
                    if j in mapping.values():
                        for k in mapping.keys():
                            if mapping[k] == j:
                                del mapping[k]
                                break

    for mapped_m1, unmapped_m1_list in external_bond_m1.items():
        if mapping[mapped_m1] not in external_bond_m2:
            continue
        unmapped_m2_list = external_bond_m2[mapping[mapped_m1]]
        if len(unmapped_m1_list) == len(unmapped_m2_list):
            for nitem in range(len(unmapped_m1_list)):
                mapping[unmapped_m1_list[nitem]] = unmapped_m2_list[nitem]

    return mapping


def constrained_embed(mol, core_mol, core_atom_mapping):
    num_core_map_atoms = len(core_atom_mapping)
    probe_ref_atom_map = list(
        zip(core_atom_mapping, range(num_core_map_atoms)))
    print(">>>", mol.GetNumConformers(), core_mol.GetNumConformers())
    _ = AllChem.AlignMol(mol, core_mol, atomMap=probe_ref_atom_map, prbCid=0)
    ff_property = ChemicalForceFields.MMFFGetMoleculeProperties(mol, "MMFF94s")
    ff = ChemicalForceFields.MMFFGetMoleculeForceField(mol, ff_property)
    for core_atom_idx in range(core_mol.GetNumAtoms()):
        core_atom_position = core_mol.GetConformer().GetAtomPosition(core_atom_idx)
        virtual_site_atom_idx = (
            ff.AddExtraPoint(
                core_atom_position.x,
                core_atom_position.y,
                core_atom_position.z,
                fixed=True,
            )
            - 1
        )
        ff.AddDistanceConstraint(
            virtual_site_atom_idx, core_atom_mapping[core_atom_idx], 0, 0, 100.0
        )
    ff.Initialize()
    max_minimize_iteration = 5
    for _ in range(max_minimize_iteration):
        minimize_seed = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        if minimize_seed == 0:
            break
    num_atoms = mol.GetNumAtoms()
    minimized_coords_list = [None] * num_atoms
    for atom_idx in range(num_atoms):
        minimized_coords = np.array(
            mol.GetConformer(0).GetAtomPosition(atom_idx), dtype=np.float64
        )
        minimized_coords_list[atom_idx] = minimized_coords
    # overwrite position information in mol
    for atom_idx in range(num_atoms):
        mol.GetConformer(0).SetAtomPosition(
            atom_idx, minimized_coords_list[atom_idx])
