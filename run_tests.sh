#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o xtrace

python3 -m unittest tests/test_wwz.py
python3 -m unittest tests/test_beta_wwz.py
