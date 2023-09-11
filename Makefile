.PHONY: test # these are not real files

python=./env/bin/python

all: env

env:
	conda env create -f ./environment.yaml -p ./env
	${python} -m pip install -e .

precommit:
	bash ./scripts/install_precommit.sh

python:
	${python} ${ARGS}

test:
	${python} -m pytest ./tests -p no:warnings
