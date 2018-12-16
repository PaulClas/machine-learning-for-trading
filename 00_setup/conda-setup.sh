#!/usr/bin/env bash
# follow pyenv / virtualenv instructions: https://github.com/pyenv/pyenv-installer
conda create -y -n bayes numpy scipy cython;
conda activate bayes;
conda config --add channels conda-forge;
pip install git+https://github.com/theano/theano;
pip install git+https://github.com/pymc-devs/pymc3;

