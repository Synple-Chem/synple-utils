import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles

from synutils.featurizers import AVAILABLE_FEATURIZERS, get_featurizer
from synutils.plotters import plot_projections


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="This module aims to utilize synple util functions",
        add_help=True,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path.cwd() / "data" / "data.csv",
        help="data path",
    )
    parser.add_argument(
        "--featurizer-name",
        type=str,
        choices=list(AVAILABLE_FEATURIZERS.keys()),
        default="morgan",
        help="featurizer name for the diversity picker",
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="product",
        help="column name for the smiles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "results",
        help="output directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(AVAILABLE_FEATURIZERS.keys())
    featurizer = get_featurizer(args.featurizer_name)
    # load data
    df = pd.read_csv(args.data_path)
    data = np.array([featurizer.get_feat(MolFromSmiles(sm)) for sm in df["product"]])
    # dim picker & get axes
    # axes = dim_picker.get_axis(data)
    # # plot
    # fig = plot_projections(val = axes)
    # # save
    # args.output_dir.mkdir(parents=True, exist_ok=True)
    # fig.savefig(args.output_dir/'plot.png')
