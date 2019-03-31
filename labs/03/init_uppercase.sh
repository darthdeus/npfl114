#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR

rm -rf $DIR/up/
rm -f tmp/*

runner=${1:-sge}

$DIR/../.venv/bin/bopt init \
        --param "alphabet_size:logscale_int:5:100" \
        --param "batch_size:logscale_int:2:1024" \
        --param "epochs:logscale_int:1:60" \
        --param "units:logscale_int:2:200" \
        --param "embedding:logscale_int:4:512" \
        --param "layers:int:2:10" \
        --param "window:int:1:11" \
        -C up \
        --runner $runner \
        $DIR/bopt_uppercase.sh
