#!/bin/bash

searchspace=transbench101_micro
datasets=(jigsaw class_object class_scene autoencoder)
start_seed=9000
n_seeds=10

experiment=$1

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

for dataset in "${datasets[@]}"
do
    for i in $(seq 0 $(($n_seeds - 1)))
    do
        sbatch ./scripts/cluster/zc_ensembles/run.sh $searchspace $dataset $start_seed $(($start_seed + $i)) $experiment
    done

    echo ""
done