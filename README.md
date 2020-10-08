# SeizureDetection
### Table of Contents
1. [Introduction](#introduction)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)

### Introduction
This project based on the [ResNeXt](https://github.com/facebookresearch/ResNeXt), where we modified the structure of the network
and extend it to use in the detection of seizure.
Some of the raw data is shown as follows, it's dark and hard to be recognized:
<!--![teaser](http://vllab.ucmerced.edu/wlai24/LapSRN/images/emma_text.gif)-->
<!--![Raw Data](/data_example/raw_data.gif)-->
After preprocessed, the video become brighter and contrast histogram be equalized, made it easier to be recognized.
![Preprocessed control data](/data_example/preprocessed_data.gif)
![Preprocessed case data](/data_example/case_data.gif)

## Requirements and Dependencies

Python: 3.7.x
Pytorch: 1.6.0 with cu101.
torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
Other required packages can check for the requirements.txt.

## Usage
User only need to modify  `python main.py`

## Pretrained Model Download 
```
wget 
```
