#!/bin/bash

# for window in 5 7 9; do
#   for dropout in 0.1 0.3 0.5; do
#     for alphabet in 50 80 100 150; do
#       python uppercase.py --window=$window --dropout_rate=$dropout --alphabet_size=$alphabet --decay_steps=2500 --epochs=5
#     done
#   done
# done

for dropout in 0.1 0.3 0.5; do
  python uppercase.py --dropout_rate=$dropout --decay_steps=2500 --epochs=7
done
