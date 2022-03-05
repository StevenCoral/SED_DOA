#!/bin/bash

# inference single fold
for RUN_IDX in {0..9}
do
    echo $'\nRun: '$RUN_IDX
    python main.py train --workspace=/home/lzpz6s/deep_optuna/sed_doae --feature_dir=/home/lzpz6s/deep_learning_project/datasets/features/ --feature_type=logmelgcc --audio_type=mic --task_type=doa_only --fold=3 --seed=20 --optuna=$RUN_IDX
done
