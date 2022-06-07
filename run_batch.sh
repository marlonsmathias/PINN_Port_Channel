#!/bin/bash

for n in {1..5};
do
    python3 train_model.py --nf 10000 --shuffle 0 --seed $n --dev 'cuda:1'
    python3 train_model.py --nf 10000 --shuffle 1 --seed $n --dev 'cuda:1'
done