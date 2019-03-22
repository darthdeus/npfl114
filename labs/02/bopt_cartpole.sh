#!/bin/bash

cd $HOME/personal_work_ms/npfl114/labs/02

$HOME/personal_work_ms/npfl114/labs/.venv/bin/python gym_cartpole.py $@ 2>&1 >>log_cartpole.txt
$HOME/personal_work_ms/npfl114/labs/.venv/bin/python gym_cartpole_evaluate.py gym_cartpole_model$JOB_ID.h5
