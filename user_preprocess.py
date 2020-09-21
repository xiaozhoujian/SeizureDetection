"""
This script is used to preprocess one-day videos
"""

import cv2
import os
import multiprocessing
from functools import partial
import time
import imageio


def mul_preprocess(h_dir, hour, video_file):
    video_name = video_file.split('.')[0]
    video_path = os.path.join(h_dir, video_file)
    output_path = os.path.join(h_dir, "{}_{}.mp4".format(hour, video_name))

    reader = imageio.get_reader(video_path)
    writer = imageio.get_writer(output_path, fps=reader.get_meta_data()['fps'], **{'macro_block_size': 1})
    # writer = imageio.get_writer(output_path, fps=reader.get_meta_data()['fps'])
    for _, im in enumerate(reader):
        im = cv2.resize(im, (454, 256))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        eq_1 = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        writer.append_data(eq_1)
    writer.close()
    os.remove(video_path)
    os.rename(output_path, video_path)


def video_preprocess(day_dir, mul_num=1):
    hours = os.listdir(day_dir)
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    for hour in hours:
        h_dir = os.path.join(day_dir, hour)
        minutes = os.listdir(h_dir)
        if mul_num == 1:
            for video_file in minutes:
                mul_preprocess(h_dir, hour, video_file)
        else:
            pool = multiprocessing.Pool(mul_num)
            func = partial(mul_preprocess, h_dir, hour)
            pool.map(func, minutes)
            pool.close()
            pool.join()


if __name__ == '__main__':
    start_time = time.time()
    video_preprocess("/Users/jojen/Workspace/cityU/data/test/2020-05-11")
    end_time = time.time()
    print("Cost {} seconds.".format(end_time - start_time))
