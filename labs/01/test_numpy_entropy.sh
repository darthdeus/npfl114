#!/bin/bash

for i in 1 2; do
  diff <(python numpy_entropy.py --data examples/numpy_entropy_data_$i.txt --model examples/numpy_entropy_model_$i.txt) examples/numpy_entropy_out_$i.txt
done

for i in 1 2 3 4 5; do
  echo "######### example $i #########"
  python numpy_entropy.py --data examples/numpy_entropy_data_$i.txt --model examples/numpy_entropy_model_$i.txt
done
