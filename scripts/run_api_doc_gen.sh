#!/usr/bin/env bash
# This script generates API documentation for the libwwz library and places the output in libwwz/docs.


if ! [[ -x "$(command -v pdoc3)" ]]; then
  echo 'Error: pdoc3 is not installed.' >&2
  exit 1
fi

set -o nounset
set -o errexit
set -o xtrace

cd ..

pdoc3 --html -c show_type_annotations=True -o docs libwwz
