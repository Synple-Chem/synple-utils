import numpy as np
import pytest
from synutils.dimension_pickers import (
    AVAILABLE_DIM_PICKERS,
    get_dim_picker
)


@pytest.fixture
def n_components() -> int:
    return 3


@pytest.mark.parametrize("dim_picker_name", AVAILABLE_DIM_PICKERS.keys())
def test_individual_dim_picker(
    dim_picker_name: str, n_components: int
):
    picker = get_dim_picker(dimension_picker_name=dim_picker_name, n_components=n_components)
    axes = picker.get_axis(np.random.rand(10, 10))
    assert axes.shape[1] == picker.dim()

