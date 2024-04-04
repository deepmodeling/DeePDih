import argparse
import os
import sys
from .workflow import build_fragment_library, build_gmx_parameter_lib, valid_gmx_parameter_lib, patch_gmx_top
from .preparation.resp_charge import get_resp_charge
from .settings import settings


def main():
    parser = argparse.ArgumentParser(description="DeePDih workflow")
    

def resp_cmd():
    # input: sdf
    # output: txt
    parser = argparse.ArgumentParser(description="Calculate RESP charge")
    parser.add_argument("--input", type=str, nargs="+", help="Input sdf file[s]")
    parser.add_argument("--output", help="Output txt file")
    parser.add_argument("--method", type=str, default='hf', help="QM method for RESP calculation. ['hf', 'dft']")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for psi4")
    parser.add_argument("--memory", help="Memory for psi4", default="2GB")
    args = parser.parse_args()

    from rdkit import Chem
    rdmol_list = []
    for name in args.input:
        rdmol = Chem.SDMolSupplier(name, removeHs=False, sanitize=True)[0]
        rdmol_list.append(rdmol)

    import psi4

    # set threads
    settings['resp_threads'] = args.threads
    settings['resp_memory'] = args.memory

    if args.method == 'hf':
        use_dft = False
    elif args.method == 'dft':
        use_dft = True
    else:
        raise ValueError(f"Method {args.method} is not supported.")

    resp_charge = get_resp_charge(rdmol_list[0], use_dft=use_dft)
    with open(args.output, "w") as f:
        f.write("\n".join([f"{c:16.8f}" for c in resp_charge]))