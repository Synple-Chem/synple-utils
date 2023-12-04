# Synple utils
Utility functions and helper functions for python scripts focus on cheminformatics.

## Setup
Setup python package:

    make env

If make doesn't work, manually set upt the environment:

	conda env create -f ./environment.yaml -p ./env
	./env/bin/python -m pip install -e .

Then, activate the environment:

    source activate ./env

## Available modules
### `synutils.featurisers`
This module contains functions to featurise molecules. Currently, it supports the following featurisers:
- fingerprint featurizers:
    - `morgan`
    - `morgan_count`
    - `rdkit`
    - `rdkit_count`
    - `topological_torsion`
    - `topological_torsion_count`
- physichem featurisers:
    - `rdkit_2d`
- combined featurisers:
    - `morgan_rdkit_2d`
    - `morgan_count_rdkit_2d`
### `synutils.dimension_pickers`
This module contains functions to pick dimensions from featurised molecules. Currently, it supports the following dimension pickers:
- `pca`
- `ica`
- `umap`
## Cheminoformatics useful tutorials
[TeachOpenCADD Talktorials](https://projects.volkamerlab.org/teachopencadd/all_talktorials.html): Nice serise of tutorials for cheminformatics.

## Cheminformatics useful external links
[openbabel](https://github.com/openbabel/openbabel0): Open Babel is a chemical toolbox designed to speak the many languages of chemical data.

    obabel -icdx input.cdx -osmi -O output.smi
