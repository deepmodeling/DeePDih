# DeePDih
 Deep Potential driven dihedral scan toolkit.

## Installation

### Requirements
- Python 3.8 or later (3.10 recommended)
- PyTorch 2
- DeePMD-kit (at least 3.0.0a0)
- networkx
- RDKit
- ASE
- Geometric
- DPData
- OpenMM (for adding bias and reading Gromacs topology)
- tblite (for semi-empirical level geometry optimization)
- Psi4 (for high-level quantum mechanical single point energy)

### Install requirements using conda

Users can install the required packages by running the following command:

```bash
conda install -c conda-forge rdkit geometric tblite-python psi4 openmm
```

### Install requirements from PyPI

```bash
pip install ase dpdata networkx
```

### Install requirements from source

#### DeePMd-kit

Please look at DeePMD-kit's [installation guide](https://github.com/deepmodeling/deepmd-kit/releases/tag/v3.0.0a0)

#### Geometric-jit

Accelerate Geometric using Numba and JIT. Please look at [this repository](https://github.com/WangXinyan940/geomeTRIC_jit).

### Install DeePDih module from source

```bash
pip install .
```

