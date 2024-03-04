# prepare gmx top for ligand
from rdkit import Chem
import subprocess
from pathlib import Path
import tempfile
import shutil
try:
    import parmed
except ImportError:
    print("Parmed is not installed. Cannot build gmx top file automatically.")
from ..utils import write_sdf


def build_gmx_top(rdmol: Chem.rdchem.Mol, top: str = "MOL_GMX.top", gro: str = "MOL_GMX.gro"):
    ncharge = Chem.GetFormalCharge(rdmol)
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        inp_sdf = tmpdir / "input.sdf"
        write_sdf(rdmol, inp_sdf)
        ret = subprocess.run(
            [
                "acpype",
                "-i",
                "input.sdf",
                "-c",
                "bcc",
                "-n",
                str(ncharge),
                "-o",
                "gmx",
                "-b",
                "MOL",
            ],
            cwd=tmpdir
        )
        shutil.copy(tmpdir / "MOL.acpype" / "MOL_GMX.gro", gro)
        system = parmed.load_file(str(tmpdir / "MOL.acpype" / "MOL_GMX.top"))
        system.save(top, overwrite=True)
    # check if we have a file with name output_top
    # if not, raise error
    outfile = Path(top)
    if not outfile.exists():
        raise RuntimeError(f"Failed to generate {top}")
