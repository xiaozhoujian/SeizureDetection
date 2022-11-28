"""
For HMDB51 and UCF101 datasets:

Code extracts frames from video at a rate of 25fps and scaling the
larger dimension of the frame is scaled to 256 pixels.
After extraction of all frames write a "done" file to signify proper completion
of frame extraction.

Usage:
  python extract_frames.py video_dir frame_dir

  video_dir => path of video files
  frame_dir => path of extracted jpg frames

"""

import os
import subprocess
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def _extract(frame_dir, cls, redo, vid_dir, vid_name):
    out_dir = os.path.join(frame_dir, cls, vid_name[:-4])
    # Checking if frames already extracted
    if os.path.isfile(os.path.join(out_dir, 'done')) and not redo:
        return
    try:
        os.system('mkdir -p "%s"' % out_dir)
        # check if horizontal or vertical scaling factor
        o = subprocess.check_output('ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 "%s"'
                                    % (os.path.join(vid_dir, cls, vid_name)), shell=True).decode('utf-8')
        lines = o.splitlines()
        width = int(lines[0].split('=')[1])
        height = int(lines[1].split('=')[1])
        resize_str = '-1:256' if width > height else '256:-1'

        # extract frames
        os.system('ffmpeg -i "%s" -r 10 -q:v 2 -vf "scale=%s" "%s"  > /dev/null 2>&1'
                  % (os.path.join(vid_dir, cls, vid_name), resize_str, os.path.join(out_dir, '%05d.jpg')))
        frame_count = len([file_name for file_name in os.listdir(out_dir)
                           if file_name.endswith('.jpg') and len(file_name) == 9])
        if frame_count == 0:
            raise Exception

        os.system('touch "%s"' % (os.path.join(out_dir, 'done')))
    except:
        print("ERROR", cls, vid_name)


def extract(vid_dir, frame_dir, redo=True):
    class_list = sorted(os.listdir(vid_dir))
    print("Classes =", class_list)
    for ic, cls in enumerate(class_list):
        vid_list = sorted(os.listdir(os.path.join(vid_dir, cls)))
        print(ic+1, len(class_list), cls, len(vid_list))
        partial_extract = partial(_extract, frame_dir, cls, redo, vid_dir)
        with Pool() as p:
            list(tqdm(p.imap(partial_extract, vid_list)))


def main():
    vid_dir = '/media/ntk/WD_BLACK_3/videos_K25'
    frame_dir = '/media/ntk/WD_BLACK_3/K25_frames'
    extract(vid_dir, frame_dir, redo=True)


if __name__ == '__main__':
    main()
