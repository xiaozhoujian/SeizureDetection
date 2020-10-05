from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import imutils
from utils import get_path_leaf


def get_test_video_online(config, video_path):
    """
    Generate list of frames according to the config parameters and video path
    :param config: config parser
    :param video_path: string, the path to a specific video
    :return: list, a list of frames that each of the element contains sample_duration frames.
    """
    clip = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_duration = config.getint('Network', 'sample_duration')
    sample_size = config.getint('Network', 'sample_size')
    if total_frames < sample_duration:
        while len(clip) < sample_duration:
            grabbed, frame = cap.read()
            if grabbed:
                frame = imutils.resize(frame, height=sample_size)
                clip.append(frame)
            else:
                # set the starting frame index 0
                cap.set(1, 0)
    else:
        while len(clip) < total_frames:
            grabbed, frame = cap.read()
            if grabbed:
                frame = imutils.resize(frame, height=sample_size)
                clip.append(frame)

    pro_clip = cv2.dnn.blobFromImages(clip, 1.0, (sample_size, sample_size),
                                      (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
    pro_clip = np.transpose(pro_clip, (1, 0, 2, 3))

    return pro_clip


class MiceOnline(Dataset):
    """MICE Dataset"""
    def __init__(self, train, config, file_list_path):
        """
        MiceOnline initial function
        :param train: int, 0 for testing, 1 for training, 2 for validation
        :param config: ConfigParser, contained the needed parameters
        :param file_list_path: string, the path of the file list, each line with the format of <video_path> #<class_id>
        :returns tuple, (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.train_val_test = train
        self.config = config
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        annotation_path = os.path.join(cur_dir, config.get('Path', 'annotation_path'))
        with open(annotation_path) as lab_file:
            self.lab_names = [line.strip('\n').split(' ')[1] for line in lab_file]

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 2

        # (filename , lab_id)
        self.data = []

        f = open(file_list_path, 'r')

        for line in f:
            video_path, class_id = line.strip('\n').split(' #')
            if os.path.exists(video_path):
                self.data.append((video_path, class_id))
            else:
                print('ERROR no such video name {}'.format(video_path))
        f.close()

    def __len__(self):
        """
        returns number of test set
        """
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label_id = self.data[idx]
        label_id = int(label_id)

        if self.train_val_test == 0:
            clip = get_test_video_online(self.config, video_path)
            video_name = get_path_leaf(video_path)
            return clip, label_id, video_name
