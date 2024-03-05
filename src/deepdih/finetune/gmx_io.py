import numpy as np
from rdkit import Chem
from typing import List, Tuple, Dict
from collections import OrderedDict

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from .paramopt import Parameters
from ..utils import TorEmbeddedMolecule
from ..utils.embedding import if_same_embed


def load_gmx(inp_top: str):
    with open(inp_top, "r") as f:
        info = []
        key = None
        for line in f:
            if len(line.strip()) == 0:
                continue
            if line.startswith(";"):
                continue
            elif line.startswith("["):
                key = line.strip()
                info.append({})
                info[-1]["key"] = key.strip()[1:-1].strip().lower()
                info[-1]["data"] = []
                continue
            info[-1]["data"].append(line.strip())
    return info


def write_gmx(info, filename):
    with open(filename, "w") as f:
        for ii in range(len(info)):
            f.write(f'[ {info[ii]["key"]} ]\n')
            for line in info[ii]["data"]:
                f.write(line+"\n")
            f.write("\n")


def load_real_indices(top: str):
    gmx_top = app.GromacsTopFile(top)
    system = gmx_top.createSystem(removeCMMotion=False)

    nparticles = system.getNumParticles()
    natoms = 0
    real_indices = []
    for ii in range(nparticles):
        isVSite = False
        try:
            system.getVirtualSite(ii)
            isVSite = True
        except mm.OpenMMException as e:
            pass
        if not isVSite:
            real_indices.append(natoms)
            natoms += 1
    return np.array(real_indices)


def update_gmx_top(rdmol: Chem.rdchem.Mol, inp_top: str, parameters: Parameters, out_top: str):
    embedded_mol = TorEmbeddedMolecule(rdmol)
    info = load_gmx(inp_top)
    real_indices = load_real_indices(inp_top)

    torsion_keys = []
    for torsion in embedded_mol.torsions:
        ii, jj, kk, ll = torsion.torsion
        ii, jj, kk, ll = real_indices[ii], real_indices[jj], real_indices[kk], real_indices[ll]
        if jj > kk:
            ii, jj, kk, ll = ll, kk, jj, ii
        torsion_keys.append((ii, jj, kk, ll))

    # work on info
    for term in range(len(info)):
        if info[term]["key"] == "dihedrals":
            tor_data = info[term]["data"]
            cleaned_data = {}
            for line in tor_data:
                line = line.split()
                key = tuple([int(i)-1 for i in line[:4]])
                val = [int(line[4]), float(line[5]), float(line[6]), int(line[7])]
                if key[1] > key[2]:
                    key = (key[3], key[2], key[1], key[0])
                if key in torsion_keys:
                    continue
                if key not in cleaned_data:
                    cleaned_data[key] = []
                cleaned_data[key].append(val)

    for torsion in embedded_mol.torsions:
        try:
            added_prm = parameters[torsion.embed]
        except ValueError as e:
            continue
        # update prm to info
        ii, jj, kk, ll = torsion.torsion
        ii, jj, kk, ll = real_indices[ii], real_indices[jj], real_indices[kk], real_indices[ll]
        if jj > kk:
            ii, jj, kk, ll = ll, kk, jj, ii
        
        added_lines = []
        for order in range(6):
            prm_val = added_prm[order].item()
            if abs(prm_val) < 1e-4:
                continue
            if prm_val > 0.0:
                added_lines.append((9, 0.0, prm_val, order+1))
            else:
                added_lines.append((9, 180.0, -prm_val, order+1))
        if len(added_lines) > 0:
            cleaned_data[(ii, jj, kk, ll)] = added_lines
        
    tor_text = []
    for key in cleaned_data:
        for val in cleaned_data[key]:
            tor_text.append(f'{key[0]+1:>5} {key[1]+1:>5} {key[2]+1:>5} {key[3]+1:>5} {val[0]:>5} {val[1]:6.2f} {val[2]:>16.8f} {val[3]:5}')

    for term in range(len(info)):
        if info[term]["key"] == "dihedrals":
            info[term]["data"] = tor_text

    write_gmx(info, out_top)
