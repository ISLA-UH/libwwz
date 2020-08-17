#!/usr/bin/env bash
# Run unit tests provided for the libwwz codebase.

set -o nounset
set -o errexit
set -o xtrace

cd ..

python3 -m unittest tests/test_wwz.py
python3 -m unittest tests/test_beta_wwz.py
