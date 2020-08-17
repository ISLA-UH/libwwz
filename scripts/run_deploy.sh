#!/usr/bin/env bash
# This script builds and deploys the libwwz library to PyPi.  A valid PyPi username and password are required.


USER=${1}
PASS=${2}

if [[ -z ${USER} ]]; then
    echo "usage: ./run_deploy.sh <username> <password>"  >&2
    exit 1
fi

if [[ -z ${PASS} ]]; then
    echo "usage: ./run_deploy.sh <username> <password>"  >&2
    exit 1
fi

if ! [[ -x "$(command -v twine)" ]]; then
  echo 'Error: twine is not installed.' >&2
  exit 1
fi

set -o nounset
set -o errexit
set -o xtrace

cd ..

python3 setup.py sdist bdist_wheel
twine upload -r pypi --username ${USER} --password ${PASS} dist/*
