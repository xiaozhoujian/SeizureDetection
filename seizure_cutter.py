# Cut the more active case using the activity of the video base on the frame different
# This file is deprecated now

import cv2
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocessing import Pool
import math


def frame_normalization(frame):
    frame = frame[128:720, 0:1280]
    frame = cv2.resize(frame, (500, 231))
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
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


def get_activity(vid_path):
    vidcap = cv2.VideoCapture(vid_path)
    success, prev_frame = vidcap.read()
    y_values = []
    while success:
        success, cur_frame = vidcap.read()
        if not success:
            break
        dif_cnt = get_diff_count(frame_normalization(prev_frame), frame_normalization(cur_frame))
        prev_frame = cur_frame
        y_values.append(dif_cnt)
    return y_values


def draw_activities(vid_path, out_path):
    y_values = get_activity(vid_path)
    mean = np.round(np.mean(np.array(y_values)), 2)
    stddev = np.round(np.std(np.array(y_values)), 2)
    plt.title("Mean: {}, Standard Deviation: {}".format(mean, stddev))
    plt.plot(range(len(y_values)), y_values)
    plt.savefig(out_path)
    plt.clf()


def draw_diff(raw_path, cut_path, out_path):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Activities before/after cut')
    raw_y = get_activity(raw_path)
    cut_y = get_activity(cut_path)
    ax1.plot(range(len(raw_y)), raw_y)
    ax2.plot(range(len(cut_y)), cut_y)

    plt.savefig(out_path)
    plt.clf()


def cut_video(vid_path, out_path):
    if os.path.exists(out_path):
        return
    frames = []
    activities = []
    raw_clips = VideoFileClip(vid_path)
    is_first_frame = True
    for cur_frame in raw_clips.iter_frames():
        if is_first_frame:
            prev_frame = cur_frame
            frames.append(prev_frame)
            is_first_frame = False
        else:
            dif_cnt = get_diff_count(frame_normalization(prev_frame), frame_normalization(cur_frame))
            prev_frame = cur_frame
            frames.append(prev_frame)
            activities.append(dif_cnt)
    raw_clips.close()
    activities = np.array(activities)
    act_mean = np.mean(activities)
    threshold = act_mean
    test_duration = int(len(activities) * 0.1)
    find_start = False
    i = 0
    end_idx = len(activities) - 1
    start_idx = 0
    test_end = len(activities) - 1
    while i < len(activities):
        if not find_start and activities[i] >= threshold:
            start_idx = i
            find_start = True
        # if the activity is lower than threshold
        if find_start and activities[i] <= threshold:
            duration_mean = np.mean(activities[i:test_end])

            if duration_mean <= threshold / 2:
                end_idx = i
                break
            else:
                i += math.ceil(test_duration / 2)
                continue
        i += 1
    out_clips = []
    for i in range(start_idx, end_idx + 1):
        out_clips.append(frames[i])
    out_clips = ImageSequenceClip(out_clips, fps=raw_clips.fps)
    out_clips.write_videofile(out_path)
    out_clips.close()


def main():
    # cut videos
    src_dir = "/root/5T/code/dataset/mice/dataset_10_3_raw"
    vid_dir = os.path.join(src_dir, "pre_case")
    out_dir = os.path.join(src_dir, "cut_case")
    plots_dir = os.path.join(src_dir, "plots")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    params = []
    print("state,video")
    for vid_file in os.listdir(vid_dir):
        if vid_file.endswith("mp4"):
            vid_path = os.path.join(vid_dir, vid_file)
            out_path = os.path.join(out_dir, vid_file)
            params.append((vid_path, out_path))
    with Pool(processes=4) as pool:
        pool.starmap(cut_video, params)
    print("Finish!")
    # draw plots of cut videos
    params = []
    for vid_file in os.listdir(out_dir):
        if vid_file.endswith("mp4"):
            raw_path = os.path.join(vid_dir, vid_file)
            cut_path = os.path.join(out_dir, vid_file)
            out_path = os.path.join(plots_dir, ''.join(vid_file.split('.')[:-1]) + '.png')
            params.append((raw_path, cut_path, out_path))
    with Pool(processes=8) as pool:
        pool.starmap(draw_diff, params)


if __name__ == '__main__':
    main()
