from __future__ import division
from torch.utils.data import Dataset
import os
import glob
import cv2
import imutils
import numpy as np
from PIL import Image
from dataset.preprocess_data import scale_crop


def video2frames(opt, video_path):
    """
    Convert video to normalized frames
    :param opt: ArgumentParser, contains config options
    :param video_path: string, video path
    :return:
        list(frames): list of all video frames
    """

    clip = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if opt.modality != 'RGB':
        raise ValueError("Config option modality not RGB, it's not support now.")
    if frame_count < opt.sample_duration:
        # frame count smaller than sample duration, loop it until clip length meet the requirement of duration
        while len(clip) < opt.sample_duration:
            ret, frame = cap.read()
            if ret:
                frame = imutils.resize(frame, width=opt.sample_size, height=opt.sample_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                clip.append(frame)
            else:
                # set the starting frame index 0
                cap.set(1, 0)
    else:
        # Convert all the frames to clip if frame count larger than sample duration, which mean contains multiple clips
        while len(clip) < frame_count:
            ret, frame = cap.read()
            if ret:
                frame = imutils.resize(frame, width=opt.sample_size, height=opt.sample_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                clip.append(frame)

    pro_clip = cv2.dnn.blobFromImages(clip, 1.0,
                                      (opt.sample_size, opt.sample_size), (114.7748, 107.7354, 99.4750),
                                      swapRB=True, crop=True)
    pro_clip = np.transpose(pro_clip, (1, 0, 2, 3))

    return pro_clip


def load_frames(opt, frame_dir, frame_count, is_train=False):
    """
    Load pre-extracted frames from specific directory
    :param opt: ArgumentParser, contains config options
    :param frame_dir: string, frame directory
    :param frame_count: int, number of frames to use under the directory
    :param is_train: boolean, indicating whether these frames are used to train network.
        If yes, return random selected frames; else return all frames
    :return:
        list(frames): list of video frames
    """

    clip = []
    i = 0
    loop = True if frame_count < opt.sample_duration else False
    if is_train:
        start_frame = 0 if frame_count < opt.sample_duration \
            else np.random.randint(0, frame_count - opt.sample_duration)
        clip_len = opt.sample_duration
    else:
        start_frame = 0
        clip_len = max(opt.sample_duration, frame_count)
    if opt.modality != 'RGB':
        raise ValueError("Config option modality not RGB, it's not support now.")
    while len(clip) < clip_len:
        frame_path = os.path.join(frame_dir, '%05d.jpg' % (start_frame + i + 1))
        try:
            im = Image.open(frame_path)
            clip.append(im.copy())
            im.close()
        except IOError:
            print('ERROR no such image {}'.format(frame_path))
        i += 1
        if loop and i == frame_count:
            # frame count smaller than sample duration, loop it until clip length meet the requirement of duration
            i = 0
    return clip


class MICE(Dataset):
    """MICE Dataset"""

    def __init__(self, dataset_type, opt):
        """
        Initialize MICE object. (tensor(frames), class_id ): Shape of tensor C x T x H x W
        :param dataset_type: int, 0 for testing, 1 for training, 2 for validation case class, 3 for validate control class
        :param opt: ArgumentParser, contains config options
        """
        self.dataset_type = dataset_type
        self.opt = opt

        with open(os.path.join(self.opt.annotation_path, "class.txt")) as lab_file:
            self.lab_names = [line.strip('\n').split(' ')[1] for line in lab_file]

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 2

        # indexes for training/test set # (filename , lab_id)
        self.data = []
        if self.dataset_type == 0:
            ann_file = opt.test_file
            file_dir = opt.frame_dir
        elif self.dataset_type == 1:
            ann_file = opt.train_file
            file_dir = opt.frame_dir
        elif self.dataset_type == 2:
            ann_file = opt.val_file_1
            file_dir = opt.val_path_1
        elif self.dataset_type == 3:
            ann_file = opt.val_file_2
            file_dir = opt.val_path_2
        else:
            raise ValueError("Dataset type must belong to 0, 1, 2, 3 which corresponding to test, train,"
                             " validate(case), validation(control)")

        f = open(os.path.join(self.opt.annotation_path, ann_file), 'r')

        for line in f:
            file_name, class_id = line.strip('\n').split(' #')
            if self.dataset_type == 1:
                file_path = os.path.join(file_dir, file_name)
            else:
                file_path = os.path.join(file_dir, file_name + '.mp4')
            if os.path.exists(file_path):
                self.data.append((file_path, class_id))
            else:
                print('{} file not exist'.format(file_path))
        f.close()

    def __len__(self):
        """
        Retrieve the len of dataloader
        :return: int, returns number of test set
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve the data in specific index
        :param idx: int, index of data
        :return: (frames, label_id, video_name)
        """
        video = self.data[idx]
        label_id = int(video[1])
        frame_path = video[0]
        frame_count = len(glob.glob(glob.escape(frame_path) + '/0*.jpg'))

        if self.dataset_type == 0:
            clip = load_frames(self.opt, frame_path, frame_count)
            video_name = frame_path.strip().split('/')[-1]
            return scale_crop(clip, self.dataset_type, self.opt), label_id, video_name
        elif self.dataset_type == 1:
            clip = load_frames(self.opt, frame_path, frame_count, is_train=True)
            video_name = frame_path.strip().split('/')[-1]
            return scale_crop(clip, self.dataset_type, self.opt), label_id, video_name
        elif self.dataset_type == 2 or self.dataset_type == 3:
            video_path = frame_path
            video_name = video_path.strip().split("/")[-1]
            clip = video2frames(self.opt, video_path)
            return clip, label_id, video_name
