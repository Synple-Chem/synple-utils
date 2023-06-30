.PHONY: test # these are not real files

all: env

env:
	conda env create -f ./environment.yaml -p ./env
	./env/bin/python -m pip install -e .

precommit:
	bash ./scripts/install_precommit.sh

python:
	./env/bin/python

test: python
	python -m pytest ./tests
