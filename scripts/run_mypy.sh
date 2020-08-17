#!/usr/bin/env bash
# Run the mypy type checker over the libwwz source.

if ! [[ -x "$(command -v mypy)" ]]; then
  echo 'Error: mypy is not installed.' >&2
  exit 1
fi

cd ..

set -o nounset
set -o errexit
set -o xtrace

mypy --config-file=.mypy.ini -m libwwz
