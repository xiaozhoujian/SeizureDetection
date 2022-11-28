# SeizureDetection
### Table of Contents
1. [Introduction](#introduction)
2. [Requirements and Dependencies](#requirements-and-dependencies)
3. [Usage](#Usage)

### Introduction
We proposed an action recognition network to detect the seizure of mice by exploring whether it's
epilepsy or not. Our network based on the [ResNeXt](https://github.com/facebookresearch/ResNeXt), where we modified the structure of the network
and extend it to use in the detection of seizure. 
We also develop post process to improve our network performance.

The raw data is shown as follows.
![Raw Data](./data_example/raw_data.gif)

## Requirements and Dependencies
Please check `environment.yml` for details.

## Usage
Download repository
```
$ git clone https://github.com/xiaozhoujian/SeizureDetection.git
```
Retrieve the kinetics 64 frames  [pretrained model](https://drive.google.com/file/d/1Cag9Zr0HvJzWRy839qgaBAMmj1b-TdcX/view). 
Put it to the `pretrained_model` directory under the project.

Before running the scripts, be sure you have downloaded the pretrained model and installed all requirements
```
$ cd SeizureDetection
$ conda env create -f environment.yml
$ conda activate seizure_detection
$ sh scripts/train.sh  # Or online_test.sh according to your need
```
Then modify the shell script `train.sh` according to your requirements. 

## Documentation
### data annotation files
Involved parameter `--train_file`, `--val_file_1`, `--val_file_2`, which are representing the label file of different videos.
Among them, the `--val_file_1` is the case data and `--val_file_2` is control data normally.
Format of annotation files shown as follows
```
<relative_path> #<label>
```
Relative directory should be given in parameter `--frame_dir`, `--val_path_1`, `--val_path_2` respectively.
Be careful, the train data should use the frame format. In this case, you can use the `extract_frames.py` to extract frames from videos.
### Result Structure
```
results/K25/
├── MICE
│   ├── MICE_train_clip64modelresnext101.log
│   ├── MICE_val_1_clip64modelresnext101.log
│   └── MICE_val_2_clip64modelresnext101.log
├── best.pth
├── save_<epoch_num>.pth
└── test_result
    ├── K25_train_epoch_0.csv
    ├── K25_val_case_epoch_<epoch_num>.csv
    └── K25_val_control_epoch_<epoch_num>.csv
```
Log files under `MICE` is the epoch-related results of train/validate. csv files under `test_result` is the probability
result on train/validate in every single video or train sample. Besides, the model weights `save_<epoch_num>.pth` will
save every 5 epochs automatically. And the epoch has the best results in training will be save to `best.pth`.