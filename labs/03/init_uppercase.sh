#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR

rm -f tmp/*

runner=${1:-sge}

results_dir=${2:-up}

if [ -d $results_dir ]; then
  echo "Directory $results_dir already exists, remove it first."
  exit 1
fi

$DIR/../.venv/bin/bopt init \
        --param "alphabet_size:logscale_int:5:100" \
        --param "batch_size:logscale_int:128:4096" \
        --param "epochs:logscale_int:3:10" \
        --param "units:logscale_int:2:200" \
        --param "embedding:logscale_int:4:512" \
        --param "layers:int:2:10" \
        --param "dropout:float:0:1" \
        --param "window:int:1:11" \
        -C $results_dir \
        --qsub=-q --qsub=cpu-troja.q \
        --qsub=-pe --qsub=smp --qsub=8 \
        --runner $runner \
        $DIR/bopt_uppercase.sh
        # --qsub=-l --qsub="mem_free=8G,act_mem_free=8G,h_vmem=12G" \
