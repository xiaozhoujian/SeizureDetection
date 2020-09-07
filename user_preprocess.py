"""
This script is used to preprocess one-day videos
"""

import cv2
import os
import multiprocessing
from functools import partial
import time


def mul_preprocess(h_dir, hour, fourcc, video_file):
    video_name = video_file.split('.')[0]
    video_path = os.path.join(h_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = os.path.join(h_dir, "{}_{}.mp4".format(hour, video_name))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        # ret返回布尔量
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eq = cv2.equalizeHist(gray)
            eq_1 = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
            out.write(eq_1)
        else:
            break
    out.release()
    cap.release()
    os.remove(video_path)
    os.rename(output_path, video_path)


def video_preprocess(day_dir, mul_num=1):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    hours = os.listdir(day_dir)
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    for hour in hours:
        h_dir = os.path.join(day_dir, hour)
        minutes = os.listdir(h_dir)
        pool = multiprocessing.Pool(mul_num)
        func = partial(mul_preprocess, h_dir, hour, fourcc)
        pool.map(func, minutes)
        pool.close()
        pool.join()


if __name__ == '__main__':
    start_time = time.time()
    video_preprocess("/Users/jojen/Workspace/cityU/data/test/2020-05-11")
    end_time = time.time()
    print("Cost {} seconds.".format(end_time - start_time))
