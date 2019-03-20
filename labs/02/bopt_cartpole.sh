#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR
# cd $HOME/personal_work_ms/npfl114/labs/02

# $HOME/personal_work_ms/npfl114/labs/.venv/bin/python gym_cartpole.py $@ 2>&1 >>log_cartpole.txt
# $HOME/personal_work_ms/npfl114/labs/.venv/bin/python gym_cartpole_evaluate.py gym_cartpole_model$JOB_ID.h5

../.cpu-venv/bin/python gym_cartpole.py $@ 2>&1 >>log_cartpole.txt
../.cpu-venv/bin/python gym_cartpole_evaluate.py 2>>log_cartpole.txt

