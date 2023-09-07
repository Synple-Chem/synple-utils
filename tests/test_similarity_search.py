from typing import List

import numpy as np
import pandas as pd
from pytest import fixture

from synutils.similarity_search import (
    chunk_similarity_search,
    numba_tanimoto,
    transform_to_fp,
    transform_to_mol,
)


@fixture
def clean_fp(clean_smiles_list: List[str]) -> np.ndarray:
    return np.array(
        [transform_to_fp(transform_to_mol(smiles)) for smiles in clean_smiles_list]
    )


def test_numba_tanimoto(clean_fp: np.ndarray):
    assert all(
        [
            numba_tanimoto(clean_fp[ii], clean_fp)[ii] == 1.0
            for ii in range(clean_fp.shape[0])
        ]
    )


def test_chunk_similarity_search(
    small_lib_df: pd.DataFrame, hit_candidates_df: pd.DataFrame
):
    max_sims, argmax_sims = chunk_similarity_search(
        chunk=small_lib_df, candidates=hit_candidates_df
    )
    assert max_sims.shape[0] == hit_candidates_df.shape[0]
    assert argmax_sims.shape[0] == hit_candidates_df.shape[0]
    in_lib_smiles_idx = [
        ii
        for ii, smiles in enumerate(hit_candidates_df["smiles"])
        if smiles in small_lib_df["product"].values
    ]
    assert all([max_sims[ii] == 1.0 for ii in in_lib_smiles_idx])
