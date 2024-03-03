import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem, ChemicalForceFields


def dihedral(i, j, k, l):
    b1, b2, b3 = j - i, k - j, l - k

    c1 = np.cross(b2, b3)
    c2 = np.cross(b1, b2)

    p1 = (b1 * c1).sum(-1)
    p1 = p1 * np.sqrt((b2 * b2).sum(-1))
    p2 = (c1 * c2).sum(-1)

    r = np.arctan2(p1, p2)
    return r / np.pi * 180.0


def angle(i, j, k):
    b1, b2 = i - j, k - j
    b1 = b1 / np.linalg.norm(b1)
    b2 = b2 / np.linalg.norm(b2)

    c1 = np.dot(b1, b2)
    r = np.arccos(c1)
    return r


def bond(i, j):
    b1 = i - j
    r = np.linalg.norm(b1)
    return r


def calc_rmsd(pos1, pos2):
    """Calculate the RMSD between two sets of positions.

    Parameters
    ----------
    pos1 : np.ndarray
        The first set of positions.
    pos2 : np.ndarray
        The second set of positions.

    Returns
    -------
    float
        The RMSD between the two sets of positions.
    """
    # Align pos1 and pos2
    # trans
    pos1 = pos1 - np.mean(pos1, axis=0)
    pos2 = pos2 - np.mean(pos2, axis=0)
    # rot
    correlation_matrix = np.dot(np.transpose(pos1), pos2)
    V, S, W_tr = np.linalg.svd(correlation_matrix)
    is_reflection = (np.linalg.det(V) * np.linalg.det(W_tr)) < 0.0
    if is_reflection:
        V[:, -1] = -V[:, -1]
    rotation = np.dot(V, W_tr)

    pos1 = np.dot(pos1, rotation)
    return np.sqrt(np.mean(np.sum((pos1 - pos2) ** 2, axis=1)))


def calc_max_disp(pos1, pos2):
    """Calculate the RMSD between two sets of positions.

    Parameters
    ----------
    pos1 : np.ndarray
        The first set of positions.
    pos2 : np.ndarray
        The second set of positions.

    Returns
    -------
    float
        The RMSD between the two sets of positions.
    """
    # Align pos1 and pos2
    # trans
    pos1 = pos1 - np.mean(pos1, axis=0)
    pos2 = pos2 - np.mean(pos2, axis=0)
    # rot
    correlation_matrix = np.dot(np.transpose(pos1), pos2)
    V, S, W_tr = np.linalg.svd(correlation_matrix)
    is_reflection = (np.linalg.det(V) * np.linalg.det(W_tr)) < 0.0
    if is_reflection:
        V[:, -1] = -V[:, -1]
    rotation = np.dot(V, W_tr)

    pos1 = np.dot(pos1, rotation)
    return np.sqrt(np.max(np.sum((pos1 - pos2) ** 2, axis=1)))


def get_mol_with_indices(
    mol_input, selected_indices=[], keep_properties=[], keep_mol_properties=[]
):
    mol_property_dict = {}
    for mol_property_name in keep_mol_properties:
        mol_property_dict[mol_property_name] = mol_input.GetProp(
            mol_property_name)
    atom_list, bond_list, idx_map = [], [], {}  # idx_map: {old: new}
    for atom in mol_input.GetAtoms():
        props = {}
        for property_name in keep_properties:
            if property_name in atom.GetPropsAsDict():
                props[property_name] = atom.GetPropsAsDict()[property_name]
        symbol = atom.GetSymbol()
        if symbol.startswith("*"):
            atom_symbol = "*"
            props["molAtomMapNumber"] = atom.GetAtomMapNum()
        elif symbol.startswith("R"):
            atom_symbol = "*"
            if len(symbol) > 1:
                atom_map_num = int(symbol[1:])
            else:
                atom_map_num = atom.GetAtomMapNum()
            props["molAtomMapNumber"] = atom_map_num
            props["dummyLabel"] = "R" + str(atom_map_num)
            props["_MolFileRLabel"] = str(atom_map_num)
        else:
            atom_symbol = symbol
        atom_list.append(
            (
                atom_symbol,
                atom.GetChiralTag(),
                atom.GetFormalCharge(),
                atom.GetNumExplicitHs(),
                props,
            )
        )
    for bond in mol_input.GetBonds():
        bond_list.append(
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
        )
    mol = Chem.RWMol(Chem.Mol())
    new_idx = 0
    for atom_index, atom_info in enumerate(atom_list):
        if atom_index in selected_indices:
            atom = Chem.Atom(atom_info[0])
            atom.SetChiralTag(atom_info[1])
            atom.SetFormalCharge(atom_info[2])
            atom.SetNumExplicitHs(atom_info[3])
            for property_name in atom_info[4]:
                if isinstance(atom_info[4][property_name], str):
                    atom.SetProp(property_name, atom_info[4][property_name])
                elif isinstance(atom_info[4][property_name], int):
                    atom.SetIntProp(property_name, atom_info[4][property_name])
            mol.AddAtom(atom)
            idx_map[atom_index] = new_idx
            new_idx += 1
    for bond_info in bond_list:
        if bond_info[0] in selected_indices and bond_info[1] in selected_indices:
            mol.AddBond(idx_map[bond_info[0]],
                        idx_map[bond_info[1]], bond_info[2])
        else:
            one_in = False
            if (bond_info[0] not in selected_indices) and (
                bond_info[1] in selected_indices
            ):
                keep_index = bond_info[1]
                remove_index = bond_info[0]
                one_in = True
            elif (bond_info[1] not in selected_indices) and (
                bond_info[0] in selected_indices
            ):
                keep_index = bond_info[0]
                remove_index = bond_info[1]
                one_in = True
            if one_in:
                if atom_list[keep_index][0] in ["N", "P"]:
                    old_num_explicit_Hs = mol.GetAtomWithIdx(
                        idx_map[keep_index]
                    ).GetNumExplicitHs()
                    mol.GetAtomWithIdx(idx_map[keep_index]).SetNumExplicitHs(
                        old_num_explicit_Hs + 1
                    )
    mol = Chem.Mol(mol)
    for mol_property_name in mol_property_dict:
        mol.SetProp(mol_property_name, mol_property_dict[mol_property_name])
    Chem.GetSymmSSSR(mol)
    mol.UpdatePropertyCache(strict=False)
    # update positions from mol_input
    conf = Chem.Conformer(mol.GetNumAtoms())
    for nidx, mol_inp_idx in enumerate(selected_indices):
        conf.SetAtomPosition(
            nidx, mol_input.GetConformer().GetAtomPosition(mol_inp_idx)
        )
    mol.AddConformer(conf)
    return mol


