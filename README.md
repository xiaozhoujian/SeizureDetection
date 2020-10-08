# SeizureDetection
### Table of Contents
1. [Introduction](#introduction)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Usage](#Usage)

### Introduction
We proposed an action recognition network to explicitly detect the seizure of mice by exploring whether it's
epilepsy or not. Our network based on the [ResNeXt](https://github.com/facebookresearch/ResNeXt), where we modified the structure of the network
and extend it to use in the detection of seizure. 
We also develop a preprocess and post process to improve the our network performance.

Some of the raw data is shown as follows, it's dark and hard to be recognized:

![Raw Data](/data_example/raw_data.gif)

After preprocessed, the video become brighter and contrast histogram be equalized, made it easier to be recognized.

![Preprocessed control data](/data_example/preprocessed_data.gif)
![Preprocessed case data](/data_example/case_data.gif)

At the same time, there are still some noise data that we filtered and won't be use to train or test
in ours network. These noise data shown as follow:

![Noise data 1](/data_example/noise_data1.gif)
![Noise data 2](/data_example/noise_data2.gif)

## Requirements and Dependencies
- Cross platform (Our scripts use at Mac, Linux or Windows system.)
- Python: 3.7.x
- CUDA 10.1 & CUDNN
- PyTorch: 1.6.0 with cu101
- NVIDIA GPU (We use GTX 2080 Super)
- Other required packages listed in the requirements.txt.

## Usage
Download repository
```
$ git clone https://github.com/xiaozhoujian/SeizureDetection.git
```
Retrieve [pretrained model](https://drive.google.com/uc?id=15nMFpl7hYT6YsnBaRzjUli6jsc6VFW77&export=download). 
Put it to the results_mice_resnext101 folder under the project.

Before running the scripts, be sure you have downloaded the pretrained model and installed all requirements
```
$ cd SeizureDetection
$ cat requirements.txt | xargs -n 1 pip install 
```
Finally, modify the section \[Path, source_dir\] of the `config.ini` according to your need.
Then run `python main.py`

