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
        --param "dropout:float:0:1" \
        --param "batch_size:logscale_int:32:1024" \
        --param "epochs:logscale_int:3:20" \
        --param "decay:discrete:exponential:polynomial" \
        --param "learning_rate:logscale_float:0.00001:0.01" \
        --param "learning_rate_final:logscale_float:0.00001:0.01" \
        --param "momentum:logscale_float:0.00001:0.01" \
        -C $results_dir \
        --qsub=-q --qsub=cpu-troja.q \
        --qsub=-pe --qsub=smp --qsub=8 \
        --runner $runner \
        $DIR/bopt_cifar.sh
        # --qsub=-l --qsub="mem_free=8G,act_mem_free=8G,h_vmem=12G" \
        # --param "optimizer:discrete:SGD:Adam" \
