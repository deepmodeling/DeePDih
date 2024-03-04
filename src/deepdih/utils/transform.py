from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw, AllChem, ChemicalForceFields
from ase import Atoms
import geometric
import geometric.molecule
import networkx as nx
import numpy as np
from typing import List


def rdmol2atoms(rdmol: Chem.rdchem.Mol) -> Atoms:
    """
    Convert an RDKit molecule to an ASE Atoms object.

    Args:
        rdmol: RDKit molecule object.

    Returns:
        ASE Atoms object.
    """

    atomic_numbers: List[int] = [a.GetAtomicNum() for a in rdmol.GetAtoms()]
    ase_atoms: Atoms = Atoms(
        numbers=atomic_numbers, positions=rdmol.GetConformers()[
            0].GetPositions()
    )
    ase_atoms.set_initial_charges([a.GetFormalCharge()
                                  for a in rdmol.GetAtoms()])
    return ase_atoms


def rdmol2mol(rdmol: Chem.rdchem.Mol):
    molecule = geometric.molecule.Molecule()
    molecule.elem = [a.GetSymbol() for a in rdmol.GetAtoms()]
    molecule.xyzs = [rdmol.GetConformer().GetPositions()]  # In Angstrom
    molecule.bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in rdmol.GetBonds()
    ]
    molecule.top_settings["read_bonds"] = True
    return molecule


def rdmol2graph(rdmol: Chem.rdchem.Mol, bond_to_break=[]) -> nx.Graph:
    atoms = [a for a in rdmol.GetAtoms()]
    bonds = [b for b in rdmol.GetBonds()]
    g = nx.Graph()
    for na, a in enumerate(atoms):
        g.add_node(a.GetIdx(), index=a.GetIdx())
    for nb, b in enumerate(bonds):
        to_break = False
        for break_bond in bond_to_break:
            if b.GetBeginAtomIdx() in break_bond and b.GetEndAtomIdx() in break_bond:
                to_break = True
                break
        if to_break:
            continue
        g.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    return g
