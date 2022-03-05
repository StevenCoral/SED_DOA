# Deep Learning Project
By Kisufit Bandach 313520074 and Idan Timsit 300320454.
You should install dependencies before proceeding.

## Prepare Dataset
The dataset can be downloaded from http://dcase.community/challenge2019/task-sound-event-localization-and-detection.  
To extract the features from audio, insert the correct input and output folders into script 1_extract_features.sh and execute.  
Let it run for some time...  

## Run Optuna Scan
Run main.py using the following line (update arguments as needed):  
python main.py train --workspace= (workspace) --feature\_dir=(features location) --feature\_type=logmelgcc --audio\_type=mic --task\_type=sed_only --fold=1 --seed=10  

Copy path of the best results output .pth file and paste instead of "pretrained\_path".  
Run a DOA training for it using the above line, but with --task\_type=doa_only.  

