#!/usr/bin/env bash

if ! [[ -x "$(command -v pdoc3)" ]]; then
  echo 'Error: pdoc3 is not installed.' >&2
  exit 1
fi

set -o nounset
set -o errexit
set -o xtrace

cd ..

pdoc3 --html -c show_type_annotations=True -o docs libwwz
