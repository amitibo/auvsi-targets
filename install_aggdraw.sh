#!/usr/bin/env bash

AGGDRAW_GIT="https://github.com/jakul/aggdraw.git"
DOWNLOAD_DIR="./aggdraw"

if [[ "$VIRTUAL_ENV" == "" ]]
then
    echo "Please run the script under virtualenv"
    exit 1
fi

git clone "$AGGDRAW_GIT" "$DOWNLOAD_DIR"
(cd "$DOWNLOAD_DIR" && python setup.py build_ext -i && python setup.py install)

