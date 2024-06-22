from rdkit import Chem
from pathlib import Path
from typing import List
import json
from .fragment import Fragmentation
from ..utils import TorEmbeddedMolecule, DeePDihError
from ..utils import (
    regularize_aromaticity,
    map_mol1_to_mol2,
    get_mol_with_indices,
    constrained_embed,
)


def get_fragments(rdmol: Chem.rdchem.Mol) -> List[Chem.rdchem.Mol]:
    frags = Fragmentation(rdmol, "MOL").get_all_fragments()
    if len(frags) == 0:
        print("No fragments found.")
        return []

    frags_return = []
    for nfrag, frag in enumerate(frags):
        piece_to_fragment = map_mol1_to_mol2(rdmol, frag)
        # try:
        #     update_positions_v2(rdmol, frag, piece_to_fragment)
        # except BaseException as e:
        #     print(f"Update positions failed: {e}")
        #     # generate conformation
        #     AllChem.EmbedMolecule(frag, randomSeed=10)
        # AllChem.EmbedMolecule(frag, randomSeed=10)
        core = get_mol_with_indices(
            rdmol, selected_indices=piece_to_fragment.keys())
        constrained_embed(frag, core, [i for i in piece_to_fragment.values()])
        frags_return.append(frag)
    return frags_return


def run_create_lib_worker(args):
    print(">>> multiprocessing start")
    f, rdmol = args
    smi = Chem.MolToSmiles(f)
    piece_to_fragment = map_mol1_to_mol2(rdmol, f)
    core = get_mol_with_indices(
        rdmol, selected_indices=piece_to_fragment.keys())
    constrained_embed(f, core, [i for i in piece_to_fragment.values()])
    print(">>> multiprocessing end")
    return smi, f


def create_lib(rdmols: List[Chem.rdchem.Mol]) -> List[Chem.rdchem.Mol]:
    input_smis = [Chem.MolToSmiles(m) for m in rdmols]
    frags, frag_smiles = [], []

    from joblib import Parallel, delayed

    input_args = []
    for nmol, rdmol in enumerate(rdmols):
        try:
            for f in Fragmentation(rdmol, "MOL").get_all_fragments():
                input_args.append((f, rdmol))
        except Exception as e:
            print(f"Mol {input_smis[nmol]} failed: {e}")
            raise e
    # detect no of cpu cores
    import os

    nproc = os.cpu_count()
    results = Parallel(n_jobs=nproc)(delayed(run_create_lib_worker)(mol) for mol in input_args)
    
    for smi, f in results:
        if smi not in frag_smiles:
            frags.append(f)
            frag_smiles.append(smi)

    # check if all the torsions are included
    emb_mols = [TorEmbeddedMolecule(m) for m in rdmols]
    emb_frags = [TorEmbeddedMolecule(frag) for frag in frags]
    for mol in emb_mols:
        for tor in mol.torsions:
            found_eqv_tor = False
            for frag in emb_frags:
                for tor_frag in frag.torsions:
                    if tor == tor_frag:
                        found_eqv_tor = True
                        break
                if found_eqv_tor:
                    break
            if not found_eqv_tor:
                raise DeePDihError(
                    f"Missing {tor} in the fragment library.")

    return frags
