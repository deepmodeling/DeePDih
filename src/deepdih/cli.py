import argparse
import os
import sys
from .workflow import build_fragment_library, build_gmx_parameter_lib, valid_gmx_parameter_lib, patch_gmx_top


def main():
    parser = argparse.ArgumentParser(description="DeePDih workflow")
    

