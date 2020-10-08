"""
This script is used to preprocess one-day videos
"""

import cv2
import os
import multiprocessing
from functools import partial
import imageio
import shutil


def mul_video(h_dir, expert, subject_num, day, hour, intermediate_dir, minute_file):
    video_path = os.path.join(h_dir, minute_file)
    minute = minute_file.strip().split('_')[-1][:2]
    dst_path = os.path.join(intermediate_dir, expert + '_' + subject_num + '_' + day +
                            '_' + hour + '_' + minute + '.mp4')
    # os.rename(video_path, dst_path)
    shutil.copy(video_path, dst_path)


def rename_video(day_dir, expert, subject_num, mul_num=1):
    """
    This function can rename the video file under a date to specific format
    :param day_dir: string, the path of the date, format like /.../2020-10-04
    :param expert: string, the name of the expert who manage the subject
    :param subject_num: string, the name of the subject, e.g. 012, c04
    :param mul_num: int, the number of multiple process, it should base on your cpu cores.
    """
    _, day = os.path.split(day_dir)
    all_dir = os.listdir(day_dir)
    hours = []
    for dir_name in all_dir:
        if dir_name.isdigit():
            hours.append(dir_name)
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    intermediate_dir = os.path.join(day_dir, 'intermediate')
    if not os.path.exists(intermediate_dir):
        os.mkdir(intermediate_dir)
    for hour in hours:
        h_dir = os.path.join(day_dir, hour)
        minutes = os.listdir(h_dir)
        func = partial(mul_video, h_dir, expert, subject_num, day, hour, intermediate_dir)
        pool = multiprocessing.Pool(mul_num)
        pool.map(func, minutes)


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

    start_frame = frame_normalization(start_frame)
    middle_frame = frame_normalization(middle_frame)
    end_frame = frame_normalization(end_frame)

    overall_diff_count = get_diff_count(start_frame, middle_frame) + get_diff_count(middle_frame, end_frame)
    if overall_diff_count > 1000:
        return True
    else:
        return False


def mul_extract(intermediate_dir, video_file):
    video_path = os.path.join(intermediate_dir, video_file)
    if not is_move(video_path):
        os.remove(video_path)


def extract_move_video(day_dir, mul_num=1):
    """
    Extract the move videos and ignore the motionless videos
    :param day_dir: string, the path of the date, format like /.../2020-10-04
    :param mul_num: int, the number of multiple process, it should base on your cpu cores.
    """
    intermediate_dir = os.path.join(day_dir, 'intermediate')
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    video_files = os.listdir(intermediate_dir)
    if mul_num == 1:
        for video_file in video_files:
            mul_extract(intermediate_dir, video_file)
    else:
        func = partial(mul_extract, intermediate_dir)
        pool = multiprocessing.Pool(mul_num)
        pool.map(func, video_files)
        pool.close()
        pool.join()


def mul_video_normalization(intermediate_dir, video_file):
    video_name = video_file.split('.')[0]
    video_path = os.path.join(intermediate_dir, video_file)
    output_path = os.path.join(intermediate_dir, "tmp_{}.mp4".format(video_name))

    reader = imageio.get_reader(video_path)
    writer = imageio.get_writer(output_path, fps=reader.get_meta_data()['fps'],
                                **{'macro_block_size': 1, 'pixelformat': 'yuv444p', 'ffmpeg_log_level': 'quiet'})
    # writer = imageio.get_writer(output_path, fps=reader.get_meta_data()['fps'])
    for _, im in enumerate(reader):
        im = cv2.resize(im, (455, 256))
        # im = cv2.resize(im, (640, 368))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        eq_1 = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        writer.append_data(eq_1)
    writer.close()
    os.remove(video_path)
    os.rename(output_path, video_path)


def video_normalization(day_dir, mul_num=1):
    """
    Normalize the videos make it easier to be recognized
    """
    intermediate_dir = os.path.join(day_dir, 'intermediate')
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    video_files = os.listdir(intermediate_dir)
    if mul_num == 1:
        for video_file in video_files:
            mul_video_normalization(intermediate_dir, video_file)
    else:
        pool = multiprocessing.Pool(mul_num)
        func = partial(mul_video_normalization, intermediate_dir)
        pool.map(func, video_files)
        pool.close()
        pool.join()
