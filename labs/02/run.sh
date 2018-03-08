#!/usr/bin/env bash

python gym_cartpole.py --epochs=200 --neurons=20 --layers=3 --activation=tanh
python gym_cartpole-evaluate.py --render

#dir="results"
#mkdir -p "$dir"
#rm -f "$dir/*"
#
#for neurons in 5 20 50; do
#    for epochs in 50 100 200; do
#        for layers in 1 2 3 5 8; do
#            fname="train-n${neurons}e${epochs}l${layers}.txt"
#            python gym_cartpole.py --epochs=$epochs --neurons=$neurons --layers=$layers --activation=tanh > "$dir/$fname"
#            python gym_cartpole-evaluate.py | tee "$dir/res-$fname" | tail
#        done
#    done
#done
