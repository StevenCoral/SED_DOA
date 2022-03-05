#!/bin/bash

# Data directory
DATASET_DIR='/home/username/deep_learning_project/datasets/extracted/'

# Feature directory
FEATURE_DIR='/home/username/deep_learning_project/datasets/features/'

# feature types: 'logmelgcc' | 'logmel'
FEATURE_TYPE='logmelgcc'

# audio types: 'mic' | 'foa'
AUDIO_TYPE='mic'

# Extract Features
python utils/feature_extractor.py --dataset_dir=$DATASET_DIR --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --data_type='dev' --audio_type=$AUDIO_TYPE
