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
    args = parser.parse_args()

    from rdkit import Chem
    rdmol = Chem.SDMolSupplier(args.input)[0]
    resp_charge = get_resp_charge(rdmol)
    with open(args.output, "w") as f:
        f.write("\n".join([f"{c:16.8f}" for c in resp_charge]))