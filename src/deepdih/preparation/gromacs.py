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
from ..geomopt import optimize, recalc_energy
from ..calculators import OpenMMBiasCalculator, merge_calculators
from ..settings import settings


def build_gmx_top(
    rdmol: Chem.rdchem.Mol, 
    top: str = "MOL_GMX.top", 
    gro: str = None, 
    use_resp: bool = False,
    opt_engine = None
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
            if opt_engine is None:
                from tblite.ase import TBLite
                opt_engine = TBLite(method="GFN2-xTB")
            bias_engine = OpenMMBiasCalculator(rdmol, restraints=[], restraint_ring=False, h_bond_repulsion=True)
            sum_engine = merge_calculators(opt_engine, bias_engine)
            # deal with conformations
            confs = []
            confs.append(rdmol)
            # for nc in range(1, num_conf):
            #     new_mol = deepcopy(rdmol)
            #     AllChem.EmbedMolecule(new_mol)
            #     mol_opt = optimize(new_mol, sum_engine, freeze=[])
            #     confs.append(new_mol)
            confs = [recalc_energy(c, opt_engine) for c in confs]
            confs = sorted(confs, key=lambda x: float(x.GetProp("ENERGY")))
            lowest_e = float(confs[0].GetProp("ENERGY"))
            confs = [c for c in confs if float(c.GetProp("ENERGY")) < lowest_e + 5.0 / 23.06054]

            # rmsd_matrix = np.zeros((len(confs), len(confs)))
            # for ii in range(len(confs)):
            #     for jj in range(ii+1, len(confs)):
            #         conf_ii = confs[ii].GetConformer().GetPositions()
            #         conf_jj = confs[jj].GetConformer().GetPositions()
            #         rmsd_matrix[ii, jj] = calc_rmsd(conf_ii, conf_jj)
            #         rmsd_matrix[jj, ii] = rmsd_matrix[ii, jj]

            # conf_remove = []
            # for ii in range(len(confs)):
            #     if ii in conf_remove:
            #         continue
            #     for jj in range(ii+1, len(confs)):
            #         if jj in conf_remove:
            #             continue
            #         if rmsd_matrix[ii, jj] < 0.5:
            #             conf_remove.append(jj)
            # conf_final = [c for i, c in enumerate(confs) if i not in conf_remove]
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
