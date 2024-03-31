import argparse
import os
import sys
from .workflow import build_fragment_library, build_gmx_parameter_lib, valid_gmx_parameter_lib, patch_gmx_top
from .preparation.resp_charge import get_resp_charge


def main():
    parser = argparse.ArgumentParser(description="DeePDih workflow")
    

def resp_cmd():
    # input: sdf
    # output: txt
    parser = argparse.ArgumentParser(description="Calculate RESP charge")
    parser.add_argument("--input", help="Input sdf file")
    parser.add_argument("--output", help="Output txt file")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for psi4")
    parser.add_argument("--memory", help="Memory for psi4", default="2GB")
    args = parser.parse_args()

    from rdkit import Chem
    rdmol = Chem.SDMolSupplier(args.input, removeHs=False, sanitize=True)[0]

    import psi4

    # set threads
    psi4.set_num_threads(args.threads)
    # set memory
    psi4.set_memory(args.memory)

    resp_charge = get_resp_charge(rdmol)
    with open(args.output, "w") as f:
        f.write("\n".join([f"{c:16.8f}" for c in resp_charge]))