"""
This is for user to use the model to detect whether today has epilepsy
"""

import os
import time
from preprocess import preprocess
import configparser
from model_predict import get_pretrained_model, predict, make_list
import shutil
from utils import get_path_leaf
import re
import argparse


def workflow(day_dir, config, pj_dir):
    st = time.time()
    mul_num = config.getint('Preprocess', 'mul_num')
    expert = config.get('Preprocess', 'expert')
    subject_name = config.get('Preprocess', 'subject_name')
    output_dir = config.get('Path', 'output_dir')
    # if config.getboolean('Pipeline', 'extract_move'):
    #     print('Start extract move video')
    #     rename_video(day_dir, expert, subject_name, output_dir, mul_num)
    #     extract_move_video(output_dir, mul_num=mul_num)
    #     print('Extract move video completed.')
    if config.getboolean('Pipeline', 'preprocess'):
        print('Start preprocess video')
        preprocess(day_dir, expert, subject_name, output_dir, mul_num=mul_num)
        print('Preprocess video completed.')
    if config.getboolean('Pipeline', 'predict'):
        date = get_path_leaf(day_dir)
        test_name = '{}_{}_{}.txt'.format(expert, subject_name, date)
        file_list_path = make_list(test_name, output_dir, date)
        model = get_pretrained_model(config)
        predict(config, model, file_list_path, pj_dir, output_dir, date)
    if config.getboolean('Pipeline', 'remove_intermediate'):
        shutil.rmtree(os.path.join(output_dir, 'intermediate'))

    print('Directory {} complete processing!'.format(day_dir))
    print("Cost {} minutes".format((time.time() - st) / 60))


def update(config, args):
    if args.source_dir:
        config.set("Path", "source_dir", value=args.source_dir)
    if args.output_dir:
        config.set("Path", "output_dir", value=args.output_dir)
    if args.expert:
        config.set("Preprocess", "expert", value=args.expert)
    if args.subject:
        config.set("Preprocess", "subject_name", value=args.subject)
    if args.preprocess:
        config.set("Pipeline", "preprocess", value=str(args.preprocess))
    else:
        config.set("Pipeline", "preprocess", value=str(False))
    if args.predict:
        config.set("Pipeline", "predict", value=str(args.predict))
    else:
        config.set("Pipeline", "predict", value=str(False))


def main(args=None):
    pj_dir = os.path.dirname(os.path.realpath(__file__))
    config = configparser.ConfigParser()
    config.read(os.path.join(pj_dir, 'config.ini'))
    if args:
        update(config, args)
    source_dir = config.get('Path', 'source_dir')
    cur_dir_name = get_path_leaf(source_dir)
    if re.match('\d\d\d\d-\d\d-\d\d', cur_dir_name):
        day_dir = source_dir
        workflow(day_dir, config, pj_dir)
    else:
        sub_dirs = os.listdir(source_dir)
        day_dirs = []
        for sub_dir in sub_dirs:
            if not re.match('\d\d\d\d-\d\d-\d\d', sub_dir):
                print('Sub directory {} under {} is not a date'.format(sub_dir, source_dir))
            else:
                day_dirs.append(os.path.join(source_dir, sub_dir))
        # result_path = os.path.join(source_dir, 'result')
        # res_video_dir = os.path.join(result_path, 'res_videos')
        # if not os.path.exists(res_video_dir):
        #     os.makedirs(res_video_dir)
        for day_dir in day_dirs:
            workflow(day_dir, config, pj_dir)
            # date_video_dir = os.path.join(day_dir, 'result', 'videos')
            # res_videos = os.listdir(date_video_dir)
            # for video in res_videos:
            #     shutil.copy(os.path.join(date_video_dir, video), res_video_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Epilepsy detection')
    parser.add_argument('--source-dir', type=str, default="")
    parser.add_argument('--output-dir', type=str, default="")
    parser.add_argument('--expert', type=str, default="")
    parser.add_argument('--subject', type=str, default="")
    parser.add_argument('--preprocess', action="store_true")
    parser.add_argument('--predict', action="store_true")
    args = parser.parse_args()
    main(args)
