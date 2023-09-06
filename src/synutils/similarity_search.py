import multiprocessing as mp
from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit, prange
from rdkit.Chem import MolFromSmiles, MolToSmiles

from synutils.enum_logger import get_logger
from synutils.featurizers import get_featurizer

LOGGER = get_logger()
FEATURIZER = get_featurizer("morgan")


@njit
def tanimoto(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates tanimoto similarity for two bit vectors

    Args:
        v1 (np.ndarray): bit vector 1
        v2 (np.ndarray): bit vector 2

    Returns:
        float: tanimoto similarity
    """
    return np.bitwise_and(v1, v2).sum() / np.bitwise_or(v1, v2).sum()


@njit(parallel=True)
def numba_tanimoto(fp: np.ndarray, fps: np.ndarray) -> np.ndarray:
    """tanimoto similarity for numba

    Args:
        fp (np.ndarray): fingerprint
        fps (np.ndarray): fingerprints

    Returns:
        np.ndarray: similarity
    """
    return_tani = np.empty(fps.shape[0], np.float32)
    for i in prange(fps.shape[0]):
        return_tani[i] = tanimoto(fp, fps[i, :])
    return return_tani


def transform_to_mol(smiles):
    "MolFromSmiles wrapper for multiprocessing"
    return MolFromSmiles(smiles)


def transform_to_smiles(mol):
    "MolToSmiles wrapper for multiprocessing"
    return MolToSmiles(mol)


def transform_to_fp(mol):
    "FEATURIZER.get_feat wrapper for multiprocessing"
    return FEATURIZER.get_feat(mol)


def chunk_similarity_search(
    chunk: pd.DataFrame,
    candidates: pd.DataFrame,
    chunk_smiles_col: str = "product",
    cand_mol_col: str = "mol",
    buff_cpu: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Similarity search of hit candidates

    Args:
        chunk (pd.DataFrame): chunk of library in pd.DataFrame
        candidates (pd.DataFrame): hit candidates in pd.DataFrame
        chunk_smiles_col (str, optional): library chunk's column name of smiles.
            Defaults to "product".
        cand_mol_col (str, optional): hit candidates' column name of mol.
        buff_cpu (int, optional): number of cpu to use. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: max similarity and index of max similarity
    """
    # transform smiles to mol and fp
    with mp.Pool(processes=mp.cpu_count() - buff_cpu) as pool:
        mols = pool.map(transform_to_mol, chunk[chunk_smiles_col])
        fps = pool.map(transform_to_fp, mols)
    fps = np.array(fps)
    LOGGER.info(f"There are {candidates.shape[0]} hit candidates left")
    cand_fps = [transform_to_fp(mol) for mol in candidates[cand_mol_col]]
    # calculate tanimoto similarity
    with mp.Pool(processes=mp.cpu_count() - buff_cpu) as pool:
        similarity_results = pool.map(partial(numba_tanimoto, fps=fps), cand_fps)
    # return max similarity and index of max similarity
    max_sims = np.array([np.max(sim) for sim in similarity_results])
    argmax_sims = np.array([np.argmax(sim) for sim in similarity_results])
    return max_sims, argmax_sims
