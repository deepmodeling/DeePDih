try:
    import psi4
except ImportError:
    print("Psi4 is not installed. Cannot calculate RESP charge automatically.")
try:
    import resp
except ImportError:
    print("Resp is not installed. Cannot calculate RESP charge automatically.")
from rdkit import Chem
from ..utils.embedding import get_eqv_atoms


def get_resp_charge(rdmol: Chem.rdchem.Mol):
    eqv_atoms = get_eqv_atoms(rdmol, layers=2)
    geom_list = []
    for atom in rdmol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        if len(atom_symbol) > 1:
            atom_symbol = f"{atom_symbol[0].upper()}{atom_symbol[1:].lower()}"
        atom_pos = rdmol.GetConformer().GetAtomPosition(atom_idx)
        geom_list.append(f"{atom_symbol} {atom_pos.x} {atom_pos.y} {atom_pos.z}\n")
    geom = "".join(geom_list)
    formal_charge = Chem.GetFormalCharge(rdmol)
    mol = psi4.geometry(geom)
    mol.set_units(psi4.core.GeometryUnits.Angstrom)
    mol.set_molecular_charge(formal_charge)
    mol.set_multiplicity(1)
    mol.update_geometry()
    mol.set_name('conformer1')

    options = {}
    mol_list = [mol]
    charges3_1 = resp.resp(mol_list, options)

    options = {}
    resp.set_stage2_constraint(mol, charges3_1[1], options)

    # options['constraint_group'] = []
    # eqv_sets = []
    # for s in eqv_atoms:
    #     new_set = set(s)
    #     if len(new_set) > 1 and new_set not in eqv_sets:
    #         eqv_sets.append(new_set)

    # for eset in eqv_sets:
    #     options['constraint_group'].append([i+1 for i in eset])

    charges3_2 = resp.resp(mol_list, options)
    charges_ret = charges3_2[1]
    for eq_grp in eqv_atoms:
        charges_ret[eq_grp] = charges_ret[eq_grp].mean()
    return charges_ret

