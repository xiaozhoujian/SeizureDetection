"""
This script is used to preprocess one-day videos
"""

import cv2
import getopt
import os, sys


video_dir = ''  # video folder absolute path
opts, args = getopt.getopt(sys.argv[1:], '-h-d:')
day_dir = ''
pj_dir = os.getcwd()
for opt_name, opt_value in opts:
    if opt_name == '-d':
        day_dir = opt_value

if day_dir == '':
    print('pleas input video directory for preprocessing.')
    sys.exit()

k = 80 # frame interval
blockSize = 16
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

h_dirs = os.listdir(day_dir)
for h in h_dirs:
    h_dir = os.path.join(day_dir, h)
    # print(h_dir)
    v_l = os.listdir(h_dir)

    for v in v_l:
        name = v[:-4]
        v_path = os.path.join(h_dir, v)
        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        prefix = h + '_'
        suffix = '.mp4'
        o_path = os.path.join(h_dir, prefix + name + suffix)
        out = cv2.VideoWriter(o_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()  # ret返回布尔量
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                eq = cv2.equalizeHist(gray)
                eq_1 = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
                out.write(eq_1)
            else:
                break
        out.release()
        cap.release()
        os.remove(v_path)