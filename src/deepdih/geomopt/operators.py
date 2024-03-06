from rdkit import Chem
from typing import List, Tuple, Dict
import numpy as np
from copy import deepcopy
from ..utils import EV_TO_HARTREE, EV_TO_KJ_MOL
from ..calculators import Calculator, merge_calculators, OpenMMBiasCalculator
from ..utils.transform import rdmol2mol, rdmol2atoms
from .optimize_tools import optimize_molecule, find_constraint_elements, rotate_position


def optimize(
    rdmol: Chem.rdchem.Mol,
    calculator: Calculator,
    freeze: List[Tuple[int]]
) -> Chem.rdchem.Mol:
    new_rdmol = deepcopy(rdmol)
    # pack rdmol to molecule
    charge = Chem.GetFormalCharge(rdmol)
    molecule = rdmol2mol(rdmol)
    # run optimization
    new_molecule = optimize_molecule(
        molecule,
        calculator,
        charge=charge,
        freeze=freeze
    )
    energies = [float(i.split()[3]) for i in new_molecule.comms]
    fin_energy = energies[-1]
    fin_conf = new_molecule.xyzs[-1]

    # update the conformation of rdmol
    for iatom in range(new_rdmol.GetNumAtoms()):
        pos = fin_conf[iatom]
        new_rdmol.GetConformer().SetAtomPosition(iatom, pos)
    # update the energy of rdmol
    new_rdmol.SetProp('ENERGY', f"{fin_energy:.12e}")
    return new_rdmol


def dihedral_scan(
    rdmol: Chem.rdchem.Mol,
    calculator: Calculator,
    torsion: Tuple[int],
    n_steps: int = 16
) -> List[Chem.rdchem.Mol]:
    torsion_delta = 360.0 / n_steps
    scan_results = []
    freeze_elements = find_constraint_elements(
        rdmol, return_all=False, add_improper=True)
    
    cons_dihedrals = find_constraint_elements(
        rdmol, return_all=True, add_improper=True)
    hbond_bias = OpenMMBiasCalculator(
        rdmol, restraints=[], h_bond_repulsion=True)
    ff_potential = merge_calculators(calculator, hbond_bias)

    # initial optimization
    rdmol = optimize(rdmol, ff_potential, freeze_elements)
    scan_results.append(rdmol)
    for istep in range(1, n_steps):
        # update conformation
        rdmol = rotate_position(rdmol, torsion, torsion_delta)
        # constraint optimization
        rdmol = optimize(rdmol, ff_potential, freeze_elements)
        # save the conformation
        scan_results.append(rdmol)
    return scan_results


def relax_conformation(rdmol: Chem.rdchem.Mol, calculator: Calculator) -> Chem.rdchem.Mol:
    # 1. add OpenMM bias forces for all the torsions we found
    res_dihedrals = find_constraint_elements(
        rdmol, return_all=True, add_improper=True)
    ff_bias = OpenMMBiasCalculator(
        rdmol, restraints=res_dihedrals, restraint_ring=True, h_bond_repulsion=True)
    ff_potential = merge_calculators(calculator, ff_bias)
    # 2. add
    freeze_elements = find_constraint_elements(
        rdmol, add_proper=True, return_all=False, add_improper=True)
    rdmol = optimize(rdmol, ff_potential, freeze_elements)
    return rdmol


def recalc_energy(
    rdmol: Chem.rdchem.Mol,
    calculator: Calculator
) -> float:
    ret_mol = deepcopy(rdmol)
    atoms = rdmol2atoms(ret_mol)
    atoms.calc = calculator
    e = atoms.get_potential_energy() * EV_TO_HARTREE
    ret_mol.SetProp('ENERGY', f"{e:.12e}")
    return ret_mol
