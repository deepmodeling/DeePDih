from rdkit import Chem
from typing import List
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
from ..settings import settings
from ..utils.tools import read_xyz_coords


def gen_multi_conformations(rdmol: Chem.Mol) -> List[Chem.Mol]:
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        Chem.MolToXYZFile(rdmol, str(tmppath / "mol.xyz"))

        formal_charge = Chem.GetFormalCharge(rdmol)
        # crest mol.xyz -T 4 -g water -chrg 0 -ewin 4 -rthr 0.2 -mquick
        subprocess.run(
            [
                "crest", 
                "mol.xyz",
                "-T", f"{settings['resp_threads']}",
                "-g", "water", "-chrg", f"{formal_charge}",
                "-ewin", "4", "-rthr", "0.2", "-squick"
            ],
            cwd=tmpdir
        )
        outfile = tmppath / "crest_conformers.xyz"
        if not outfile.exists():
            raise RuntimeError("Crest failed to generate conformations")
        coords = read_xyz_coords(outfile)

        ret_molecules = []
        for crd in coords:
            new_mol = deepcopy(rdmol)
            # set coords to new_mol
            for iatom in range(new_mol.GetNumAtoms()):
                new_mol.GetConformer().SetAtomPosition(iatom, crd[iatom])
            ret_molecules.append(new_mol)
    
    return ret_molecules
        
            