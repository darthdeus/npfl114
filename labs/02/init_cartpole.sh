#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR

rm -rf $DIR/cartpole/
rm models/*

$DIR/../.venv/bin/bopt init \
        --param "batch_size:int:2:100" \
        --param "epochs:int:1:100" \
        --param "layers:int:2:10" \
        --param "units:int:2:50" \
        -C cartpole \
        --runner local \
        $DIR/bopt_cartpole.sh
