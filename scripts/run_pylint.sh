#!/usr/bin/env bash

if ! [[ -x "$(command -v pylint)" ]]; then
  echo 'Error: pylint is not installed.' >&2
  exit 1
fi

cd ..

set -o nounset
set -o errexit
set -o xtrace

pylint libwwz
