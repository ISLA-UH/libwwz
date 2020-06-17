#!/usr/bin/env bash

if ! [[ -x "$(command -v pylint)" ]]; then
  echo 'Error: pylint is not installed.' >&2
  exit 1
fi

pylint libwwz
pylint beta_wwz
