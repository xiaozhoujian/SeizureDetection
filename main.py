"""
This is for user to use the model to detect whether today has epilepsy
"""

import os
import time
from preprocess import rename_video, extract_move_video, video_normalization
import configparser
from model_predict import get_pretrained_model, predict, make_list
import shutil
from utils import get_path_leaf


def main():
    st = time.time()
    pj_dir = os.path.dirname(os.path.realpath(__file__))
    config = configparser.ConfigParser()
    config.read(os.path.join(pj_dir, 'config.ini'))
    day_dir = config.get('Path', 'source_dir')

    # if day_dir.split()
    mul_num = config.getint('Preprocess', 'mul_num')
    expert = config.get('Preprocess', 'expert')
    subject_name = config.get('Preprocess', 'subject_name')
    if config.getboolean('Pipeline', 'extract_move'):
        print('Start extract move video')
        rename_video(day_dir, expert, subject_name, mul_num)
        extract_move_video(day_dir, mul_num=mul_num)
        print('Extract move video completed.')
    if config.getboolean('Pipeline', 'preprocess'):
        print('Start normalize video')
        video_normalization(day_dir, mul_num=12)
        print('Normalize video completed.')
    if config.getboolean('Pipeline', 'predict'):
        date = get_pathleaf(day_dir)
        test_name = '{}_{}_{}.txt'.format(expert, subject_name, date)
        file_list_path = make_list(test_name, day_dir)
        model = get_pretrained_model(config)
        predict(config, model, file_list_path, pj_dir)
    if config.getboolean('Pipeline', 'remove_intermediate'):
        shutil.rmtree(os.path.join(day_dir, 'intermediate'))

    print('Complete processing!')
    print("Cost {} minutes".format((time.time() - st)/60))


if __name__ == '__main__':
    main()

