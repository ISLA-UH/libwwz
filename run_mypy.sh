#!/usr/bin/env bash

if ! [[ -x "$(command -v mypy)" ]]; then
  echo 'Error: mypy is not installed.' >&2
  exit 1
fi

mypy --config-file=.mypy.ini -m libwwz
mypy --config-file=.mypy.ini -m beta_wwz
