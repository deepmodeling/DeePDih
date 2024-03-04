import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from rdkit import Chem
from ..utils.geometry import calc_max_disp, calc_rmsd, calc_r2


def plot_opt_results(opt_confs: List[Chem.rdchem.Mol], ref_confs: List[Chem.rdchem.Mol], filename: str = "result.png", draw_energy: bool = True) -> None:
    opt_positions = [conf.GetConformer().GetPositions() for conf in opt_confs]
    ref_positions = [conf.GetConformer().GetPositions() for conf in ref_confs]
    max_disp = [calc_max_disp(opt_pos, ref_pos) for opt_pos, ref_pos in zip(opt_positions, ref_positions)]
    rmsd = [calc_rmsd(opt_pos, ref_pos) for opt_pos, ref_pos in zip(opt_positions, ref_positions)]
    
    # subplot 1: the distribution of max displacement
    if draw_energy:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
    else:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
    plt.hist(max_disp, bins=20)
    plt.xlabel('Max displacement (Angstrom)')
    plt.ylabel('Count')
    plt.title('Max displacement distribution')
    # subplot 2: the distribution of RMSD
    if draw_energy:
        plt.subplot(1, 3, 2)
    else:
        plt.subplot(1, 2, 2)
    plt.hist(rmsd, bins=20)
    plt.xlabel('RMSD (Angstrom)')
    plt.ylabel('Count')
    plt.title('RMSD distribution')
    # subplot 3: the scatter of energy
    if draw_energy:
        plt.subplot(1, 3, 3)
        opt_energies = np.array([float(conf.GetProp('ENERGY')) for conf in opt_confs]) * 627.5
        ref_energies = np.array([float(conf.GetProp('ENERGY')) for conf in ref_confs]) * 627.5
        opt_energies = opt_energies - opt_energies[0]
        ref_energies = ref_energies - ref_energies[0]
        e_min = min(opt_energies.min(), ref_energies.min())
        e_max = max(opt_energies.max(), ref_energies.max())
        e_min = e_min - 0.1 * (e_max - e_min)
        e_max = e_max + 0.1 * (e_max - e_min)
        plt.xlim(e_min, e_max)
        plt.ylim(e_min, e_max)
        r2 = calc_r2(opt_energies, ref_energies)
        plt.scatter(opt_energies, ref_energies)
        plt.plot([e_min, e_max], [e_min, e_max], 'r--')
        plt.xlabel('Optimized energy (kcal/mol)')
        plt.ylabel('Reference energy (kcal/mol)')
        plt.title('Energy scatter. R2 = {:.4f}'.format(r2))
        plt.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
    plt.tight_layout()
    plt.savefig(filename)
