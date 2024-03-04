from typing import List, Tuple
from rdkit import Chem
import numpy as np
import deepdih
from pathlib import Path
from . import dihedral, get_torsions_from_rotamer, get_rotamers, getRingAtoms


EMBED_W1 = np.load(Path(deepdih.__file__).parent / "data/embed_w1.npy")


def mol_to_graph_matrix(rdmol: Chem.rdchem.Mol) -> Tuple[np.ndarray, np.ndarray]:
    # return adjacency matrix and node features
    # node features are one hot version of atomic numbers
    # adjacency matrix is 1 if two atoms are bonded, 0 otherwise
    # embedding: element one-hot + aromatic + degree + formal_charge + num_hydrogens
    num_atom = rdmol.GetNumAtoms()
    adj = np.zeros((num_atom, num_atom))
    node_features = np.zeros((num_atom, 128 + 4))
    all_ring_atoms = getRingAtoms(rdmol, join=True)
    for i in range(num_atom):
        adj[i, i] = 1
        # element one-hot
        node_features[i, rdmol.GetAtomWithIdx(i).GetAtomicNum()] = 1
        # saturated
        degree = rdmol.GetAtomWithIdx(i).GetTotalDegree()
        valance = rdmol.GetAtomWithIdx(i).GetExplicitValence()
        if degree == valance:
            node_features[i, 128] = 1
        else:
            node_features[i, 128] = 0
        # node_features[i, 128] = int(rdmol.GetAtomWithIdx(i).GetIsAromatic())
        # degree
        node_features[i, 129] = rdmol.GetAtomWithIdx(i).GetDegree()
        # formal charge
        node_features[i, 130] = rdmol.GetAtomWithIdx(i).GetFormalCharge()
        # get neighbors
        neighbors = rdmol.GetAtomWithIdx(i).GetNeighbors()
        # get linked atoms
        num_hs = len([atom for atom in neighbors if atom.GetAtomicNum() == 1])
        # if num_hs == 0:
        #     node_features[i, 131] = 0
        # else:
        #     node_features[i, 131] = 1
        # get if atom is on a ring
        if i in all_ring_atoms:
            node_features[i, 131] = 1
        else:
            node_features[i, 131] = 0

        for j in range(num_atom):
            if rdmol.GetBondBetweenAtoms(i, j) is not None:
                adj[i, j] = 1
    return adj, node_features


def get_embed(rdmol: Chem.rdchem.Mol):
    adj, node = mol_to_graph_matrix(rdmol)
    natom = adj.shape[0]
    support = np.dot(node[:, :128], EMBED_W1[:128, :128])
    out = np.dot(adj, support)
    out = np.concatenate((out, node[:, 128:]), axis=1)
    return out


def get_eqv_atoms(rdmol: Chem.rdchem.Mol):
    embed = get_embed(rdmol)
    natom, nfeat = embed.shape[0], embed.shape[1]
    dist = np.power(
        embed.reshape((natom, 1, nfeat)) - embed.reshape((1, natom, nfeat)), 2
    ).sum(axis=2)
    eqv_list = []
    for na in range(natom):
        eqv_list.append([na])
        for nb in range(natom):
            if dist[na, nb] < 1e-2 and na != nb:
                eqv_list[-1].append(nb)
    return eqv_list



def if_same_embed(e1, e2):
    dist1 = np.linalg.norm(e1 - e2)
    dist2 = np.linalg.norm(e1[::-1, :] - e2)
    dist = min(dist1, dist2)
    if dist < 1e-2:
        return True
    return False

class EmbeddedTorsion:
    def __init__(self, torsion, embed, param=None):
        self.torsion = torsion
        self.embed = embed
        self.param = param

    def __repr__(self) -> str:
        if self.param is not None:
            return f"Embedded torsion ({self.torsion[0]},{self.torsion[1]},{self.torsion[2]},{self.torsion[3]}) with parameter idx {self.param}."
        return f"Embedded torsion ({self.torsion[0]},{self.torsion[1]},{self.torsion[2]},{self.torsion[3]})."

    def __eq__(self, __value: object) -> bool:
        return if_same_embed(self.embed, __value.embed)


class TorEmbeddedMolecule:
    def __init__(self, rdmol: Chem.rdchem.Mol, conf=[], target=[]):
        self.rdmol = rdmol
        self.smiles = Chem.MolToSmiles(rdmol) 
        self.torsions = []
        embed = get_embed(rdmol)
        all_tors = []
        embed = get_embed(rdmol)
        for tor in get_rotamers(rdmol):
            all_tors += get_torsions_from_rotamer(rdmol, tor[1:3])
        for tor in all_tors:
            tor_embed = embed[tor, :]
            self.torsions.append(EmbeddedTorsion(tor, tor_embed))
        self.conf = conf
        self.target = target
        self.tor_vals = []
        self.calc_tor_of_confs()

    def update_conf(self, conf, target):
        self.conf = conf
        self.target = target
        self.calc_tor_of_confs()

    def calc_tor_of_confs(self):
        vals = []
        for conf in self.conf:
            vals.append([])
            for tor in self.torsions:
                vals[-1].append(
                    dihedral(
                        conf[tor.torsion[0]],
                        conf[tor.torsion[1]],
                        conf[tor.torsion[2]],
                        conf[tor.torsion[3]],
                    )
                )
        vals = np.array(vals)
        self.tor_vals = vals

    def __repr__(self) -> str:
        return f"Molecule with {len(self.torsions)} torsions and {len(self.conf)} data points."


def if_torsions_all_same(mol1: TorEmbeddedMolecule, mol2: TorEmbeddedMolecule):
    if len(mol1.torsions) != len(mol2.torsions):
        return False
    for tor1 in mol1.torsions:
        found = True
        for tor2 in mol2.torsions:
            if tor1 == tor2:
                found = True
                break
        if not found:
            return False
    return True


def deduplicate_rdmols(mol_list: List[Chem.rdchem.Mol]):
    embed_mols = [TorEmbeddedMolecule(m) for m in mol_list]
    deduplicated_mols = []
    for mol in embed_mols:
        found = False
        for n_dedupl in range(len(deduplicated_mols)):
            # if the torsions are the same, choose the larger one
            if if_torsions_all_same(mol, deduplicated_mols[n_dedupl]):
                found = True
                if (
                    mol.rdmol.GetNumAtoms()
                    > deduplicated_mols[n_dedupl].rdmol.GetNumAtoms()
                ):
                    deduplicated_mols[n_dedupl] = mol
                break
        if not found:
            deduplicated_mols.append(mol)
    return [i.rdmol for i in deduplicated_mols]
