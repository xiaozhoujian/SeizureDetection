import cv2
import os
import imutils
import shutil
import time
from functools import partial
from itertools import product
import multiprocessing


def mul_video(h_dir, expert, subject_num, day, hour, minute_file):
    video_path = os.path.join(h_dir, minute_file)
    dst_path = os.path.join(h_dir, expert + '_' + subject_num + '_' + day + '_' + hour + '_' + minute_file)
    os.rename(video_path, dst_path)


def rename_video(day_dir, expert, subject_num, mul_num=1):
    _, day = os.path.split(day_dir)
    hours = os.listdir(day_dir)
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    for hour in hours:
        h_dir = os.path.join(day_dir, hour)
        minutes = os.listdir(h_dir)
        func = partial(mul_video, h_dir, expert, subject_num, day, hour)
        pool = multiprocessing.Pool(mul_num)
        pool.map(func, minutes)


def frame_normalization(frame):
    frame = frame[128:720, 0:1280]
    frame = imutils.resize(frame, width=500)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # normalizes the brightness and increases the contrast of the image.
    frame = cv2.equalizeHist(frame)
    return frame


def get_diff_count(frame1, frame2):
    diff_array = cv2.absdiff(frame1, frame2)
    # Blurs an image using the median filter, used to remove noise from an image or signal
    diff_array = cv2.medianBlur(diff_array, 3)

    diff_array = cv2.threshold(diff_array, 40, 255, cv2.THRESH_BINARY)[1]

    diff_count = cv2.countNonZero(diff_array)
    return diff_count


def is_move(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    _, start_frame = cap.read()

    cap.set(cv2.CAP_PROP_POS_FRAMES, int((frame_count - 1 - 1) / 2))
    _, middle_frame = cap.read()

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    _, end_frame = cap.read()
    cap.release()

    start_frame = frame_normalization(start_frame)
    middle_frame = frame_normalization(middle_frame)
    end_frame = frame_normalization(end_frame)

    overall_diff_count = get_diff_count(start_frame, middle_frame) + get_diff_count(middle_frame, end_frame)
    if overall_diff_count > 1000:
        return True
    else:
        return False


def mul_extract(h_dir, out_dir, video_file):
    video_path = os.path.join(h_dir, video_file)
    if is_move(video_path):
        shutil.copy(video_path, out_dir)


def extract_move_video(day_dir, remove_source=False, mul_num=1):
    hours = os.listdir(day_dir)
    out_dir = os.path.join(day_dir, 'move')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    for hour in hours:
        h_dir = os.path.join(day_dir, hour)
        minutes = os.listdir(h_dir)
        func = partial(mul_extract, h_dir, out_dir)
        pool = multiprocessing.Pool(mul_num)
        pool.map(func, minutes)
        pool.close()
        pool.join()
        if remove_source:
            shutil.rmtree(h_dir, ignore_errors=True)


def main(day_dir, expert, subject_name):
    rename_video(day_dir, expert, subject_name)
    extract_move_video(day_dir)


if __name__ == '__main__':
    start_time = time.time()
    extract_move_video("/Users/jojen/Workspace/cityU/data/test/2020-05-11")
    end_time = time.time()
    print("Cost {} seconds.".format(end_time - start_time))
