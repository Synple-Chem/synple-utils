from typing import List

import numpy as np
import pytest
from rdkit.Chem import MolFromSmiles

from synutils.featurizers import (
    AVAILABLE_FEATURIZERS,
    Featurizer,
    FingerprintFeaturizer,
    MultiFeaturizer,
    get_featurizer,
)


@pytest.fixture
def fp_size() -> int:
    return 2048


@pytest.mark.parametrize("featurizer_name", AVAILABLE_FEATURIZERS.keys())
def test_individual_featurizer(
    clean_smiles_list: List[str], featurizer_name: str, fp_size: int
):
    featurizer = get_featurizer(featurizer_name=featurizer_name)
    feat = featurizer.get_feat(MolFromSmiles(clean_smiles_list[0]))
    assert feat.ndim == 1
    assert len(feat) == featurizer.dim()

    if isinstance(featurizer, FingerprintFeaturizer):
        if not featurizer.count:
            assert np.max(feat) == 1


def test_multi_featurizer(clean_smiles_list: List[str], fp_size: int):
    n_feat = 0
    featurizers: List[Featurizer] = []
    for featurizer_name in AVAILABLE_FEATURIZERS.keys():
        featurizer = get_featurizer(featurizer_name)
        n_feat += featurizer.dim()
        featurizers.append(featurizer)

    multifeaturizer = MultiFeaturizer(featurizers)
    feats = multifeaturizer.get_feat(MolFromSmiles(clean_smiles_list[0]))
    assert feats.ndim == 1
    assert len(feats) == n_feat
