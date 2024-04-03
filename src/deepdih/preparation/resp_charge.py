try:
    import psi4
except ImportError:
    print("Psi4 is not installed. Cannot calculate RESP charge automatically.")
try:
    import resp
except ImportError:
    print("Resp is not installed. Cannot calculate RESP charge automatically.")
from rdkit import Chem
from typing import List
from ..utils.embedding import get_eqv_atoms


def get_resp_charge(rdmol_list: List[Chem.rdchem.Mol]):
    psi4.driver.qcdb.parker._expected_bonds = {
        'H': 1,
        'C': 4,
        'N': 3,
        'O': 2,
        'F': 1,
        'P': 3,
        'S': 2,
        "CL": 1,
        "BR": 1,
        "I": 1
    }

    eqv_atoms = get_eqv_atoms(rdmol_list[0], layers=4)
    mol_list = []
    for nmol, rdmol in enumerate(rdmol_list):
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
        mol.set_name(f'conformer{nmol}')
        mol_list.append(mol)

    options = {
        "VDW_RADII": {
            "BR": 1.80,
            "I": 1.95
        },
        "BASIS_ESP": "def2-SVP"
    }
    charges3_1 = resp.resp(mol_list, options)

    options = {
        "VDW_RADII": {
            "BR": 1.80,
            "I": 1.95
        },
        "BASIS_ESP": "def2-SVP"
    }
    resp.set_stage2_constraint(mol_list[0], charges3_1[1], options)
    options['RESP_A'] = 0.001
    
    options['grid'] = []
    options['esp'] = []

    # Add constraint for atoms fixed in second stage fit
    for structure in range(len(mol_list)):
        options['grid'].append('%i_%s_grid.dat' % (structure + 1, mol_list[structure].name()))
        options['esp'].append('%i_%s_grid_esp.dat' % (structure + 1, mol_list[structure].name()))
    
    eqv_sets = [set(atoms) for atoms in eqv_atoms]

    new_set = []
    for eqv_set in eqv_sets:
        if len(eqv_set) > 1 and eqv_set not in new_set:
            new_set.append(eqv_set)

    final_eqv_list = []
    for eqv_set in new_set:
        eqv_list = []
        for atom in eqv_set:
            eqv_list.append(atom)
        final_eqv_list.append(eqv_list)

    # options["constraint_group"] = final_eqv_list

    charges3_2 = resp.resp(mol_list, options)
    charges_ret = charges3_2[1]
    for eqv_grp in final_eqv_list:
        charges_ret[eqv_grp] = charges_ret[eqv_grp].mean()
    return charges_ret

