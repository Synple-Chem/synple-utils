# Synple utils
Utility functions that are recursively used in multiple repos.

## Setup
Setup python package:

    make

If make doesn't work, manually set upt the environment:

	conda env create -f ./environment.yaml -p ./env
	./env/bin/python -m pip install -e .

Then, activate the environment:

    source activate ./env

## Cheminoformatics useful tutorials

## Cheminformatics useful external links
[openbabel](https://github.com/openbabel/openbabel0): Open Babel is a chemical toolbox designed to speak the many languages of chemical data.

    obabel -icdx input.cdx -osmi -O output.smi
