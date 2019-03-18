#!/bin/bash

cd ~/projects/npfl114/labs/02

~/.venvs/tf2/bin/python gym_cartpole.py $@ 2>&1 >>log_cartpole.txt
~/.venvs/tf2/bin/python gym_cartpole_evaluate.py
