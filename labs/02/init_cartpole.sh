#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR

rm -rf $DIR/cartpole/
rm models/*

$DIR/../.venv/bin/bopt init \
        --param "batch_size:logscale_int:2:100" \
        --param "epochs:logscale_int:1:200" \
        --param "layers:int:2:10" \
        --param "units:logscale_int:2:200" \
        -C cartpole \
        --runner sge \
        $DIR/bopt_cartpole.sh
