#! /usr/bin/env bash

# fail automatically if any of the scripts fail
set -e

scripts/download.sh
scripts/build_rmis.sh
scripts/prepare.sh
scripts/execute.sh
