"""
This is for user to use the model to detect whether today has epilepsy
"""

import os
import time
from preprocess import preprocess
from model_predict import get_pretrained_model
from model_predict import predict
from model_predict import make_list
import shutil
from utils import get_path_leaf
import re
from opts import parse_args
import multiprocessing


def workflow(day_dir, args):
    st = time.time()
    mul_num = args.mul_num
    expert = args.expert
    subject_name = args.subject_name
    output_dir = args.output_dir
    post_process = args.post_process
    svm = args.svm
    if args.preprocess:
        print('Start preprocess video')
        preprocess(day_dir, expert, subject_name, output_dir, mul_num=mul_num)
        print('Preprocess video completed.')
    if args.predict:
        date = get_path_leaf(day_dir)
        test_name = '{}_{}_{}.txt'.format(expert, subject_name, date)
        file_list_path = make_list(test_name, output_dir, date)
        model = get_pretrained_model(args)
        print("Post process: {}, svm: {}".format(post_process, svm))
        predict(args, model, file_list_path, output_dir, date, args.result_name,
                post_process=post_process, svm=svm)
    if args.remove_intermediate:
        shutil.rmtree(os.path.join(output_dir, 'intermediate'))

    print('Directory {} complete processing!'.format(day_dir))
    print("Cost {} minutes".format((time.time() - st) / 60))


def main(args=None):
    source_dir = args.source_dir
    cur_dir_name = get_path_leaf(source_dir)
    if re.match('\d\d\d\d-\d\d-\d\d', cur_dir_name):
        day_dir = source_dir
        workflow(day_dir, args)
    else:
        sub_dirs = os.listdir(source_dir)
        day_dirs = []
        for sub_dir in sub_dirs:
            if not re.match('\d\d\d\d-\d\d-\d\d', sub_dir):
                print('Sub directory {} under {} is not a date'.format(sub_dir, source_dir))
            else:
                day_dirs.append(os.path.join(source_dir, sub_dir))
        for day_dir in day_dirs:
            workflow(day_dir, args)


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main(parse_args())
