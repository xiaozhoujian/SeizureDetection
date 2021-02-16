"""
This script is used to preprocess one-day videos
"""

import cv2
import os
import multiprocessing
from functools import partial
import imageio
from utils import get_path_leaf


def frame_normalization(frame):
    frame = frame[128:720, 0:1280]
    # frame = imutils.resize(frame, width=500)
    frame = cv2.resize(frame, (500, 231))
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

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 2)
    _, end_frame = cap.read()
    cap.release()

    try:
        start_frame = frame_normalization(start_frame)
        middle_frame = frame_normalization(middle_frame)
        end_frame = frame_normalization(end_frame)
    except:
        print(video_file)
        return False

    overall_diff_count = get_diff_count(start_frame, middle_frame) + get_diff_count(middle_frame, end_frame)
    if overall_diff_count > 1000:
        return True
    else:
        return False


def mul_preprocess(intermediate_dir, expert, subject_name, day, hour, video_path):
    if not video_path.endswith('mp4'):
        os.remove(video_path)
        return None
    if not is_move(video_path):
        return None
    minute = get_path_leaf(video_path).split('.')[0][-2:]
    preprocessed_file_path = os.path.join(intermediate_dir, "{}_{}_{}_{}_{}.mp4".format(
        expert, subject_name, day, hour, minute
    ))
    reader = imageio.get_reader(video_path)
    if reader.get_meta_data()['duration'] < 59:
        reader.close()
        os.remove(video_path)
        return None
    writer = imageio.get_writer(preprocessed_file_path, fps=reader.get_meta_data()['fps'],
                                **{'ffmpeg_log_level': 'panic', 'macro_block_size': None})
    for _, im in enumerate(reader):
        im = cv2.resize(im, (640, 360))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        eq_1 = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        writer.append_data(eq_1)
    writer.close()


def preprocess(day_dir, expert, subject_num, output_dir, mul_num=1):
    """
    This function can preprocess video and store it to the output dir
    :param day_dir: string, the path of the date, format like /.../2020-10-04
    :param expert: string, the name of the expert who manage the subject
    :param subject_num: string, the name of the subject, e.g. 012, c04
    :param output_dir: string, the path of the directory to store the intermediate files
    :param mul_num: int, the number of multiple process, it should base on your cpu cores.
    """
    _, day = os.path.split(day_dir)
    all_dir = os.listdir(day_dir)
    hours = []
    for dir_name in all_dir:
        if dir_name.isdigit():
            hours.append(dir_name)
    intermediate_dir = os.path.join(output_dir, 'intermediate', day)
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
    for hour in hours:
        h_dir = os.path.join(day_dir, hour)
        video_files = [os.path.join(h_dir, x) for x in os.listdir(h_dir)]
        if mul_num == 1:
            for video_file in video_files:
                mul_preprocess(intermediate_dir, expert, subject_num, day, hour, video_file)
        else:
            func = partial(mul_preprocess, intermediate_dir, expert, subject_num, day, hour)
            pool = multiprocessing.Pool(mul_num)
            pool.map(func, video_files)
            pool.close()
            pool.join()
