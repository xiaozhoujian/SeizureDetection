import cv2
import getopt
import os
import sys
import imutils
import shutil


def rename_video(day_dir, expert, subject_num):
    _, day = os.path.split(day_dir)
    hours = os.listdir(day_dir)
    for hour in hours:
        h_dir = os.path.join(day_dir, hour)
        minutes = os.listdir(h_dir)
        for minute_file in minutes:
            video_path = os.path.join(h_dir, minute_file)
            dst_path = os.path.join(h_dir, expert + '_' + subject_num + '_' + day + '_' + hour + '_' + minute_file)
            os.rename(video_path, dst_path)


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


def extract_move_video(day_dir):
    hours = os.listdir(day_dir)
    out_dir = os.path.join(day_dir, 'move')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for hour in hours:
        h_dir = os.path.join(day_dir, hour)
        minutes = os.listdir(h_dir)
        for video_file in minutes:
            video_path = os.path.join(h_dir, video_file)
            if is_move(video_path):
                shutil.copy(video_path, out_dir)
        shutil.rmtree(h_dir, ignore_errors=True)


def main():
    opts, _ = getopt.getopt(sys.argv[1:], '-h-d:-e:-n:')
    day_dir = expert = subject_name = ''
    for opt_name, opt_value in opts:
        if opt_name == '-d':
            day_dir = opt_value
        elif opt_name == '-e':
            expert = opt_value
        elif opt_name == '-n':
            subject_name = opt_value

    if day_dir == '':
        print('Pleas input video directory for preprocessing.')
        sys.exit()

    rename_video(day_dir, expert, subject_name)
    extract_move_video(day_dir)


if __name__ == '__main__':
    main()
