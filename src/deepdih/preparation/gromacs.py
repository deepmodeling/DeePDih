# prepare gmx top for ligand
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import subprocess
from pathlib import Path
import tempfile
import shutil
from copy import deepcopy
try:
    import parmed
except ImportError:
    print("Parmed is not installed. Cannot build gmx top file automatically.")
from ..utils import write_sdf, calc_rmsd
from .resp_charge import get_resp_charge
from .conformations import gen_multi_conformations
from ..geomopt import optimize, recalc_energy
from ..calculators import OpenMMBiasCalculator, merge_calculators
from ..settings import settings


def build_gmx_top(
    rdmol: Chem.rdchem.Mol, 
    top: str = "MOL_GMX.top", 
    gro: str = None, 
    use_resp: bool = False,
    multi_conf: bool = True
):
    ncharge = Chem.GetFormalCharge(rdmol)
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        inp_sdf = tmpdir / "input.sdf"
        write_sdf(rdmol, str(inp_sdf))
        ret = subprocess.run(
            [
                "acpype",
                "-i",
                "input.sdf",
                "-c",
                "bcc" if not use_resp else "gas",
                "-n",
                str(ncharge),
                "-o",
                "gmx",
                "-b",
                "MOL",
            ],
            cwd=tmpdir
        )
        top_file = tmpdir / "MOL.acpype" / "MOL_GMX.top"
        if not top_file.exists():
            raise RuntimeError(f"Failed to generate {top}")
        if gro is not None:
            shutil.copy(tmpdir / "MOL.acpype" / "MOL_GMX.gro", gro)

        if use_resp:
            # deal with conformations
            if multi_conf:
                confs = gen_multi_conformations(rdmol)
                if len(confs) > 8:
                    confs = confs[:8]
            else:
                confs = []
            confs.append(rdmol)
            conf_final = confs

            conf_final_sdf = []
            for i, c in enumerate(conf_final):
                conf_final_sdf.append(f"resp_conf_{i}.sdf")
                write_sdf(c, str(tmpdir / f"resp_conf_{i}.sdf"))

            ret = subprocess.run(
                [
                    "deepdih-resp",
                    "--input",
                    *conf_final_sdf,
                    "--output",
                    "charge.txt",
                    "--threads",
                    str(settings["resp_threads"]),
                    "--memory",
                    settings["resp_memory"]
                ],
                cwd=tmpdir
            )
            with open(f"/{str(tmpdir)}/charge.txt", "r") as f:
                charges = [float(line.strip()) for line in f]

        system = parmed.load_file(str(tmpdir / "MOL.acpype" / "MOL_GMX.top"))
        if use_resp:
            for natom in range(len(system.atoms)):
                system.atoms[natom].charge = charges[natom]

        system.save(top, overwrite=True)
    # check if we have a file with name output_top
    # if not, raise error
    outfile = Path(top)
    if not outfile.exists():
        raise RuntimeError(f"Failed to generate {top}")
