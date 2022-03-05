#!/bin/bash

# Data directory
DATASET_DIR='/home/lzpz6s/deep_learning_project/datasets/extracted/'
DATASET_DIR='/Users/kisufitbandach/dcase2019_task3_data'


# Feature directory
FEATURE_DIR='/home/lzpz6s/deep_learning_project/datasets/features/'
FEATURE_DIR='/Users/kisufitbandach/dcase2019_task3_features'


# Workspace
WORKSPACE='/home/lzpz6s/deep_learning_project/sed_doae/'
WORKSPACE='/Users/kisufitbandach/repos/sed_doae/'


FEATURE_TYPE='logmelgcc'
AUDIO_TYPE='mic'
SEED=10

# GPU number
GPU_ID=0

# Train SED
# TASK_TYPE: 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
TASK_TYPE='sed_only'
for FOLD in {1..4}
    do
    echo $'\nFold: '$FOLD
    CUDA_VISIBLE_DEVICES=$GPU_ID python ${WORKSPACE}main.py train --workspace=$WORKSPACE --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --fold=$FOLD --seed=$SEED
done

# Train DOA
# TASK_TYPE: 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
TASK_TYPE='doa_only'
for FOLD in {1..4}
    do
    echo $'\nFold: '$FOLD
    CUDA_VISIBLE_DEVICES=$GPU_ID python ${WORKSPACE}main.py train --workspace=$WORKSPACE --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --fold=$FOLD --seed=$SEED
done





