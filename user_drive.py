"""
this is for user to use the model and predict
input should be a directory of day
"""

import os
import time
from user_screen import rename_video, extract_move_video
from user_preprocess import video_preprocess
import configparser
import user_post
import user_predict
import multiprocessing
from functools import partial


def main():
    st = time.time()
    pj_dir = os.path.dirname(os.path.realpath(__file__))
    config = configparser.ConfigParser()
    config.read(os.path.join(pj_dir, 'config.ini'))
    day_dir = config.get('Extract Move', 'dir')
    mul_num = config.getint('Others', 'mul_num')
    if config.getboolean('Pipeline', 'extract_move'):
        expert = config.get('Extract Move', 'expert')
        subject_name = config.get('Extract Move', 'subject_name')
        print('Start extract move video')
        # extract the move video
        rename_video(day_dir, expert, subject_name, mul_num)
        extract_move_video(day_dir, remove_source=config.getboolean('Extract Move', 'remove_source'), mul_num=mul_num)
        print('Extract move video completed.')
    if config.getboolean('Pipeline', 'preprocess'):
        print('Start preprocess video')
        video_preprocess(day_dir)
        print('Preprocess video complete')
    if config.getboolean('Pipeline', 'predict'):
        hour_list = os.listdir(day_dir)
        model = user_predict.get_pretrained_model()
        sample_size = config.getint('Network', 'sample_size')
        sample_duration = config.getint('Network', 'sample_duration')
        separator = config.get('Others', 'separator')
        for hour in hour_list:
            predict_dir = os.path.join(day_dir, hour)
            print('Predicting videos in {}'.format(predict_dir))
            result_path = os.path.join(day_dir, hour + '_predict.csv')
            try:
                multiprocessing.set_start_method('spawn')
            except RuntimeError:
                pass
            video_list = [os.path.join(predict_dir, x) for x in os.listdir(predict_dir)]
            func = partial(user_predict.main, model, sample_size, sample_duration)
            pool = multiprocessing.Pool(mul_num)
            results = pool.map(func, video_list)
            pool.close()
            pool.join()
            with open(result_path, 'w') as f:
                for result in results:
                    video_name = result[0].split(separator)[-1]
                    f.write("{},{}\n".format(video_name, result[1]))
            # for file in video_list:
            #     user_predict.main(inputs=os.path.join(predict_dir, file),  model=model,
            #                       sample_size=sample_size, sample_duration=sample_duration)
            print('-------------------------------------------------------------\n')

        user_post.main(day_dir)

    et = time.time()
    used_t = (et-st)/60
    print('Complete Prediction!')
    print("Cost {} minutes".format(used_t))


if __name__ == '__main__':
    main()

