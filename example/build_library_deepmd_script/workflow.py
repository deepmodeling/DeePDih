import os
import numpy as np
import deepdih


def main():
    print("Prepare Calculator")
    # load QM calculator
    from dp_calculator import DPCalculator
    qm_calc = DPCalculator("model.pt")

    # set rerun calculators
    rerun_calc = None
    # from ase.calculators.psi4 import Psi4
    # rerun_calc = Psi4(method="wb97x-d", basis="def2-svp", memory="2GB", num_threads=4)

    # load molecule
    mol_files = os.listdir("molecules")

    print("Build Fragment Library")
    deepdih.workflow.build_fragment_library(
        [f"molecules/{mol}" for mol in mol_files],
        qm_calc,
        recalc_calculator=rerun_calc
    )

    print("Build GMX Parameter Library")
    deepdih.workflow.build_gmx_parameter_lib(parameter_lib="param.pkl")

    print("Valid GMX Parameter Library")
    deepdih.workflow.valid_gmx_parameter_lib(parameter_lib="param.pkl")

    print("Patch GMX Topology")
    if not os.path.exists("molecules_patched"):
        os.makedirs("molecules_patched")
    for mol_name in mol_files:
        name = mol_name.split(".")[0]
        deepdih.workflow.patch_gmx_top(
            f"molecules/{mol_name}",
            "param.pkl",
            "tmp.top",
            f"molecules_patched/{name}.top"
        )
        os.remove("tmp.top")


if __name__ == "__main__":
    main()