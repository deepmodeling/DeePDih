import numpy as np
import torch
from typing import List, Tuple, Optional
from ..utils import TorEmbeddedMolecule
from ..settings import settings


class Parameters:

    def __init__(self):
        self.embeddings = []
        self.parameters = torch.zeros((0, 6))
        self._built = False

    def __len__(self):
        return len(self.embeddings)

    def __contains__(self, embedding: np.ndarray):
        for nemb, emb in enumerate(self.embeddings):
            if np.linalg.norm(emb - embedding) < 1e-2:
                return True
        return False

    def __getitem__(self, embedding: np.ndarray):
        for nemb, emb in enumerate(self.embeddings):
            if np.linalg.norm(emb - embedding) < 1e-2:
                return self.parameters[nemb]
        raise ValueError(f"Embedding {embedding} not found in parameters.")

    def __repr__(self):
        return f"Parameters with {len(self.embeddings)} embeddings and parameters."

    def get_idx(self, embedding: np.ndarray):
        for nemb, emb in enumerate(self.embeddings):
            if np.linalg.norm(emb - embedding) < 1e-2:
                return nemb
        raise ValueError(f"Embedding {embedding} not found in parameters.")

    def add(self, embedding: np.ndarray):
        if embedding not in self:
            self.embeddings.append(embedding)
            prm_tmp = torch.from_numpy(np.random.randn(6)).reshape((1, 6)) * 2.
            self.parameters = torch.cat((self.parameters, prm_tmp))

    def remove(self, embedding: np.ndarray):
        if embedding in self:
            idx = self.get_idx(embedding)
            self.embeddings.pop(idx)
            self.parameters = torch.cat(
                (self.parameters[:idx], self.parameters[idx+1:]))


def initialize_parameters(molecules: List[TorEmbeddedMolecule]) -> Parameters:
    parameters = Parameters()
    for mol in molecules:
        for tor in mol.torsions:
            parameters.add(tor.embed)
    return parameters


# input 1: list of TorEmbeddedMolecule
# input 2: Tuple[np.ndarray, np.ndarray] (parameter embeddings, parameter values)
# return : list of [dihedral angles in (nconfs, ndihedrals), targets in (nconfs,), param indices in (ndihedrals,)]
def build_training_data(molecules: List[TorEmbeddedMolecule], parameters: Optional[Parameters] = None) -> Tuple[Parameters, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if parameters is None:
        parameters = initialize_parameters(molecules)

    ret_list = []
    for mol in molecules:
        tor_angles = mol.tor_vals * np.pi / 180.0
        tor_angles = torch.from_numpy(tor_angles)
        targets = np.array(mol.target).ravel()
        targets = torch.from_numpy(targets)
        tor_prm_idx = torch.tensor(
            [parameters.get_idx(tor.embed) for tor in mol.torsions])
        ret_list.append((tor_angles, targets, tor_prm_idx))
    return parameters, ret_list


def loss_molecule(param_vals: torch.Tensor, dihedrals: torch.Tensor, targets: torch.Tensor, prm_idx: torch.Tensor) -> torch.Tensor:
    param_kconsts = param_vals[prm_idx].reshape(
        (1, -1, 6))  # (1, ndihedrals, 6)
    order = torch.tensor([1, 2, 3, 4, 5, 6]).reshape((1, 1, 6))
    cos_inner = dihedrals.reshape(
        (dihedrals.shape[0], dihedrals.shape[1], 1)) * order
    cos_val = torch.cos(cos_inner)  # (nconfs, ndihedrals, 6)
    energy_term = (cos_val * param_kconsts).sum(axis=2).sum(axis=1).ravel()
    energy_term_center = energy_term - torch.mean(energy_term)
    targets_center = targets - torch.mean(targets)
    return torch.pow(energy_term_center - targets_center, 2).mean()


def optimize_parameters(parameters: Parameters, data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], l1_reg: float = 0.0, l2_reg: float = 0.1) -> Parameters:
    param_vals = torch.autograd.Variable(
        parameters.parameters, requires_grad=True)
    optimizer = torch.optim.AdamW([param_vals], lr=1.0, weight_decay=0.0)

    num_epochs = settings['optimization_steps']
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss_vals = [loss_molecule(param_vals, *datum) for datum in data]
        loss_sum = sum(loss_vals)
        loss = loss_sum / len(loss_vals)
        reg = l1_reg * torch.abs(param_vals).mean() + l2_reg * torch.sqrt(torch.pow(param_vals, 2).mean())
        loss_tot = loss + reg
        loss_tot.backward()
        optimizer.step()
        if epoch == 0 or (epoch+1) % 1000 == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs}, loss: {loss.item()}, reg: {reg.item()}")
    return parameters


def finetune_workflow(molecules: List[TorEmbeddedMolecule], n_fold: int = 3, l1_reg: float = 0.1) -> Parameters:
    param_list = []
    for nf in range(n_fold):
        params, data = build_training_data(molecules)
        params = optimize_parameters(params, data, l1_reg=l1_reg)
        param_list.append(params)

    # merge parameters
    ret_params = Parameters()
    for emb in param_list[0].embeddings:
        ret_params.add(emb)
    for emb in param_list[0].embeddings:
        prm_list = [params[emb] for params in param_list]
        prm_mean = torch.stack(prm_list).mean(axis=0)
        ret_params.parameters[ret_params.get_idx(emb)] = prm_mean
    return ret_params
