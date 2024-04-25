#!/bin/bash

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
