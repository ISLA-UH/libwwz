#!/usr/bin/env bash

# Clean build products.

set -o nounset
set -o errexit
set -o xtrace

cd ..

rm -rf .mypy_cache
rm -rf build
rm -rf dist
rm -rf libwwz.egg-info
rm -rf docs/*
