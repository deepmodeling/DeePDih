import os
import pickle
import numpy as np
from rdkit import Chem
from ase.calculators.calculator import Calculator
from typing import List, Tuple, Dict, Optional
from ..calculators.gromacs import GromacsTopCalculator
from ..preparation import build_gmx_top
from ..settings import settings
from ..geomopt import dihedral_scan, recalc_energy, relax_conformation, plot_opt_results
from ..mollib import create_lib
from ..finetune import finetune_workflow, update_gmx_top
from ..utils import (
    read_sdf, 
    write_sdf, 
    get_rotamers, 
    get_all_torsions_to_opt, 
    EV_TO_HARTREE, 
    EV_TO_KJ_MOL, 
    TorEmbeddedMolecule
)


def load_mol_files(mol_files: list) -> list:
    mols = []
    for mol_file in mol_files:
        mols_ = read_sdf(mol_file)
        mols.extend(mols_)
    return mols


def prepare_frags(mol_files: list, output_sdf: str = "fragments.sdf") -> list:
    mols = load_mol_files(mol_files)
    fragments = create_lib(mols)
    write_sdf(fragments, output_sdf)
    return fragments


def scan_frag_rotamers(
    fragment: Chem.rdchem.Mol,
    calculator: Calculator,
    output_name: Optional[str] = None,
    recalc_calculator: Optional[Calculator] = None
) -> list:
    rotamers = get_rotamers(fragment)
    dih_results = []
    for rot in rotamers:
        dih_result_rot = dihedral_scan(
            fragment, calculator, rot, settings['dihedral_scan_steps'])
        dih_results.extend(dih_result_rot)
    if recalc_calculator is not None:
        dih_results = [recalc_energy(mol, recalc_calculator)
                       for mol in dih_results]
    if output_name is not None:
        write_sdf(dih_results, output_name)
    return dih_results


def build_fragment_library(
    mol_files: list,
    calculator: Calculator,
    output_fragment_sdf: str = "fragments.sdf",
    output_folder: str = "fragments",
    recalc_calculator: Optional[Calculator] = None
) -> list:
    fragments = prepare_frags(mol_files, output_fragment_sdf)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for nfrag, frag in enumerate(fragments):
        scan_frag_rotamers(
            frag,
            calculator,
            f"{output_folder}/fragment_{nfrag}.sdf",
            recalc_calculator
        )


def build_gmx_parameter_lib(
    fragments_sdf: str = "fragments.sdf",
    lib_folder: str = "fragments",
    output_gmx_top_folder: str = "frag_top",
    output_mm_relax_folder: str = "mm_relax",
    parameter_lib: str = "param.pkl",
    plot: bool = True
):
    fragments = read_sdf(fragments_sdf)

    # 1. relax the fragments
    if not os.path.exists(output_mm_relax_folder):
        os.makedirs(output_mm_relax_folder)
    if not os.path.exists(output_gmx_top_folder):
        os.makedirs(output_gmx_top_folder)
    for nfrag, frag in enumerate(fragments):
        frag = fragments[nfrag]
        build_gmx_top(
            frag, top=f"{output_gmx_top_folder}/fragment_{nfrag}.top")
        calculator = GromacsTopCalculator(
            frag, f"{output_gmx_top_folder}/fragment_{nfrag}.top")
        init_conformations = read_sdf(f"{lib_folder}/fragment_{nfrag}.sdf")
        relax_conformations = [
            relax_conformation(
                mol, 
                calculator
            ) 
            for mol in init_conformations
        ]
        recalc_conformations = [recalc_energy(
            mol, calculator) for mol in relax_conformations]
        write_sdf(recalc_conformations,
                  f"{output_mm_relax_folder}/fragment_{nfrag}.sdf")
        if plot:
            plot_opt_results(recalc_conformations, init_conformations,
                             f"{output_mm_relax_folder}/fragment_{nfrag}_relax.png")

    # 2. build the parameter library
    # 2.1 prepare training data
    train_data = []
    for nfrag, frag in enumerate(fragments):
        qm_confs = read_sdf(f"{lib_folder}/fragment_{nfrag}.sdf")
        mm_confs = read_sdf(f"{output_mm_relax_folder}/fragment_{nfrag}.sdf")

        opt_tors = get_all_torsions_to_opt(frag)
        rerun_calculator = GromacsTopCalculator(
            frag, f"{output_gmx_top_folder}/fragment_{nfrag}.top", turnoff_propers=opt_tors)
        recalc_confs = [recalc_energy(mol, rerun_calculator)
                        for mol in mm_confs]

        mm_positions = [conf.GetConformer().GetPositions()
                        for conf in recalc_confs]
        qm_energies = np.array([float(conf.GetProp('ENERGY'))
                               for conf in qm_confs])
        mm_energies = np.array([float(conf.GetProp('ENERGY'))
                               for conf in recalc_confs])
        qm_energies = qm_energies - qm_energies.mean()
        mm_energies = mm_energies - mm_energies.mean()
        delta_energies = qm_energies - mm_energies
        delta_energies = delta_energies / EV_TO_HARTREE * EV_TO_KJ_MOL
        embedded_mol = TorEmbeddedMolecule(
            frag, conf=mm_positions, target=delta_energies)
        train_data.append(embedded_mol)

    # 2.2 train model
    params = finetune_workflow(
        train_data, n_fold=settings['optimization_folds'])

    with open(parameter_lib, 'wb') as f:
        pickle.dump(params, f)


def valid_gmx_parameter_lib(
    fragments_sdf: str = "fragments.sdf",
    init_top_folder: str = "frag_top",
    lib_folder: str = "fragments",
    mm_relax_folder: str = "mm_relax",
    parameter_lib: str = "param.pkl",
    output_valid_folder: str = "valid",
    plot: bool = True
):
    if not os.path.exists(output_valid_folder):
        os.makedirs(output_valid_folder)

    fragments = read_sdf(fragments_sdf)
    with open(parameter_lib, 'rb') as f:
        params = pickle.load(f)

    for nfrag, frag in enumerate(fragments):
        update_gmx_top(frag, f"{init_top_folder}/fragment_{nfrag}.top",
                       params, f"{output_valid_folder}/fragment_{nfrag}.top")
        calculator = GromacsTopCalculator(
            frag, f"{output_valid_folder}/fragment_{nfrag}.top")
        init_confs = read_sdf(f"{lib_folder}/fragment_{nfrag}.sdf")
        relax_confs = read_sdf(f"{mm_relax_folder}/fragment_{nfrag}.sdf")
        recalc_confs = [recalc_energy(mol, calculator) for mol in relax_confs]
        write_sdf(recalc_confs, f"{output_valid_folder}/fragment_{nfrag}.sdf")
        if plot:
            r2, rmse = plot_opt_results(
                recalc_confs, init_confs, f"{output_valid_folder}/fragment_{nfrag}.png")
            print(f"Frag {nfrag} R2: {r2:.3f}, RMSE: {rmse:.3f}")


def patch_gmx_top(frag_mol: str, param_lib: str, input_top: str = "lig_init.top", output_top: str = "lig_mod.top"):
    frag = read_sdf(frag_mol)[0]
    with open(param_lib, 'rb') as f:
        params = pickle.load(f)

    if not os.path.exists(input_top):
        build_gmx_top(frag, input_top)

    update_gmx_top(frag, input_top, params, output_top)


def gmx_top_to_amber(gmx_top: str = "lig_mod.top", output_amber: str = "lig_mod.prmtop"):
    import parmed as pmd
    parm = pmd.load_file(gmx_top)
    parm.save(output_amber)