from typing import List

import pandas as pd
from pytest import fixture

from synutils.path import TEST_PATH


@fixture
def clean_smiles_list() -> List[str]:
    return [
        "O=C(O)c1ccccc1",
        "O=C([O-])c1ccccc1",
        "O=C(O[Na])c1ccccc1",
        "CN(C)c1ccc(C=O)cc1",
        "O=Cc1ccccc1O",
        "CCC(C=O)CC",
    ]


@fixture
def small_lib_df() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH / "resource" / "small_lib.csv")


@fixture
def hit_candidates_df() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH / "resource" / "hit_candidates.csv")


# def temp_db(temp_path:Path):

#     # TODO Create a sqlite instance in temp_path / database.sqlite
#     # TODO Use models/sqlalchemy to create database schema
#     # TODO Fill with fake data / e.g. using tables stores in tests/resources/tables.csv
#     # TODO Return instance of session for test queries

#     return session
