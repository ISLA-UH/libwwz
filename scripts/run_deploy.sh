#!/usr/bin/env bash
# This script builds and deploys the libwwz library to PyPi.  A valid PyPi username and password are required.

if [[ -z ${USER} ]]; then
    echo "usage: ./run_deploy.sh <token>"  >&2
    exit 1
fi

if ! [[ -x "$(command -v twine)" ]]; then
  echo 'Error: twine is not installed.' >&2
  exit 1
fi

TOKEN=${1}

set -o nounset
set -o errexit
set -o xtrace

cd ..

#python3 setup.py sdist bdist_wheel
python3 -m build .
twine upload -r pypi -u __token__ -p ${TOKEN} --skip-existing dist/*

VERSION="v$(python -c 'import toml; print(toml.load("pyproject.toml")["project"]["version"])')"
git tag -a ${VERSION} -m"Release ${VERSION}"
git push origin ${VERSION}


