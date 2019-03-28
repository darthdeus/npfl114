#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR

export JOB_ID=${JOB_ID:-$$}

mkdir -p tmp
test_out_fname=tmp/uppercase_test_out$JOB_ID.txt
dev_out_fname=tmp/uppercase_dev_out$JOB_ID.txt

echo "Outputs are:

dev:  $dev_out_fname
test: $test_out_fname
"

$DIR/../.venv/bin/python $DIR/uppercase.py --dev_out_fname=$dev_out_fname --test_out_fname=$test_out_fname $@ 2>&1
$DIR/../.venv/bin/python $DIR/uppercase_eval.py $dev_out_fname uppercase_data_dev.txt 2>&1
