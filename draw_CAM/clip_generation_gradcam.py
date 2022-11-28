import glob
import torch
import torchvision.transforms.functional as F
import numpy as np
import os
import cv2
import numbers
import json
from PIL import Image
from dataset.preprocess_data import mpc
import collections


def normalize(img):
    # normalize the clip
    # TODO test
    mean = np.array([114.7748, 107.7354, 99.4750])  # ActivityNet
    std = np.array([1, 1, 1])
    img_norm = (img - mean) / std
    return img_norm


def normalize_1(img):
    img = ((img / 255) - 0.5) * 2
    img_norm = np.clip(img, -1, 1)
    return img_norm


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self):
        pass


class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        pass


def generate_clip(opts):
    # total_frames = len(glob.glob(glob.escape(opts.input_path)+'/0*.jpg'))
    if '[' or ']' in opts.input_path:
        opts.input_path = opts.input_path.translate({ord('['): '[[]', ord(']'): '[]]'})
    clip_all = sorted(glob.glob(opts.input_path + r'/0*.jpg'))  # get all the frames of a video
    total_frames = len(clip_all)
    clip_pick = []
    # if out of frame range, loop it
    if total_frames < opts.clip_s + opts.clip_length:
        index = 0
        opts.clip_s = 0
    else:
        index = opts.clip_s
    while len(clip_pick) < opts.clip_length:
        clip_pick.append(clip_all[index])
        # 隔四帧取一次？
        index += 1
        if index >= total_frames:
            index = opts.clip_s

    # clip_path = clip_path[opts.clip_s: opts.clip_s+opts.clip_length] # pick a clip
    clip = []
    # input_size default is 112
    scale = Scale(opts.input_size)
    center_crop = CenterCrop(opts.input_size)

    img_ori = []
    for frame_path in clip_pick:
        img_arr_ori = Image.open(frame_path)
        # img_arr = img_arr.resize((opts.input_size, opts.input_size))

        # Center Crop
        img_arr = scale(img_arr_ori)
        img_arr = center_crop(img_arr)

        # Resize
        # img_arr = img_arr_ori.resize((opts.input_size, opts.input_size), Image.BILINEAR)

        img_center = Image.fromarray(np.uint8(img_arr))
        img_ori.append(img_center)

        img_arr_np = np.float32(img_arr)
        img_arr_np = normalize(img_arr_np)
        clip.append(img_arr_np)
    clip = np.array(clip)
    clip = clip.transpose(3, 0, 1, 2)
    clip_tensor = torch.from_numpy(clip).float()
    clip_tensor = clip_tensor.unsqueeze(0)
    clip_var = torch.autograd.Variable(clip_tensor, requires_grad=True)
    if opts.cuda:
        clip_var = clip_var.cuda()
    return clip_var, img_ori
    # return clip_var


def generate_clip_saliency(opts, agent, sp_transform, criterion):
    def array_to_tensor(clip_tensor, cuda):
        clip_var = torch.autograd.Variable(clip_tensor, requires_grad=True)  # need?
        if cuda:
            clip_var = clip_var.cuda()
        return clip_var

    clip_all = sorted(glob.glob(opts.input_path + r'/0*.jpg'))  # get all the frames of a video
    total_frames = len(clip_all)
    clip_pick = []
    # if out of frame range, loop it
    if total_frames < opts.clip_s + opts.clip_length:
        index = 0
        opts.clip_s = 0
    else:
        index = opts.clip_s
    while len(clip_pick) < opts.clip_length:
        clip_pick.append(clip_all[index])
        index += 1
        if index >= total_frames:
            index = opts.clip_s

    or_clip = torch.Tensor(1, 3, len(clip_pick), opts.height, opts.width)
    ct_clip = torch.Tensor(1, 3, len(clip_pick), opts.sample_size, opts.sample_size)
    clip_image_resize = []
    for i, frame_path in enumerate(clip_pick):
        img_arr = Image.open(frame_path)
        clip_image_resize.append(img_arr.resize((opts.input_size, opts.input_size)))
        ct_clip[:, :, i, :, :] = sp_transform[1](img_arr)
        img_ori_resize = img_arr.resize((opts.width, opts.height), Image.BILINEAR)
        img_ori_resize = sp_transform[0](img_ori_resize)
        or_clip[:, :, i, :, :] = img_ori_resize
    ct_clip_var = array_to_tensor(ct_clip, opts.cuda)
    clip_sp_var = array_to_tensor(mpc(or_clip, ct_clip, agent, opts.steps_rl, opts.clip_class, criterion), opts.cuda)

    return ct_clip_var, clip_sp_var, clip_image_resize


def generate_clip_mp(opts):
    def calculate_center(array, clip_s):
        centers = []
        for i in range(clip_s, clip_s + len(array)):
            if i > len(array) - 1:
                # if array[-1][0] != 160 or array[-1][0] != 120:
                centers.append(array[-1])  # when the index is the last frame
            else:
                # if array[i-1][0] != 160 or array[i-1][0] != 120:
                centers.append(array[i - 1])

        centers = np.array(centers)
        crop_x, crop_y = np.average(centers, axis=0)

        return crop_x, crop_y

    class DestinedCrop(object):
        def __init__(self, size, crop_x, crop_y, padding=0, pad_if_needed=False, image=False):
            if isinstance(size, numbers.Number):
                self.size = (int(size), int(size))
            else:
                self.size = size
            self.padding = padding
            self.pad_if_needed = pad_if_needed
            self.crop_x = crop_x
            self.crop_y = crop_y
            self.image = image

        def get_params(self, img, output_size):
            """Get parameters for ``crop`` for a aimed crop.

            Args:
                img (PIL Image): Image to be cropped.
                output_size (tuple): Expected output size of the crop.

            Returns:
                tuple: params (c, i, j, h, w) to be passed to ``crop`` for random crop.
            """

            w, h = img.size
            th, tw = output_size
            if self.crop_x < tw / 2:
                x = tw / 2
            elif self.crop_x > w - tw / 2:
                x = w - tw / 2
            else:
                x = self.crop_x

            if self.crop_y < th / 2:
                y = th / 2
            elif self.crop_y > h - th / 2:
                y = h - th / 2
            else:
                y = self.crop_y

            x = int(x - tw / 2)  # move the crop destination to left corner
            y = int(y - th / 2)
            return y, x, th, tw

        def __call__(self, img):
            """
            Args:
                img (PIL Image): Image to be cropped. 
            Returns:
                PIL Image: Cropped image.
            """

            i, j, h, w = self.get_params(img, self.size)

            if self.padding > 0:
                img = F.pad(img, self.padding)

            # pad the width if needed
            if self.pad_if_needed and img.size[0] < self.size[1]:
                img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
            # pad the height if needed
            if self.pad_if_needed and img.size[1] < self.size[0]:
                img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))
            img_crop = F.crop(img, i, j, h, w)
            if self.image:
                return img_crop
            else:
                img_crop_np = np.float32(img_crop)
                img_crop_np = normalize(img_crop_np)
                return img_crop_np

    def check_cropped_image(clip, image_name, crop_if=False):
        path_to_dir = '../results/{}'.format(image_name)
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)
        i = 0
        if crop_if:
            print('Saving motion patch of {}'.format(image_name))
            for img in clip:
                # img = img.transpose(1,2,0)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # cv2.imshow('image',img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                path_to_file = os.path.join(path_to_dir, image_name + '_mb_{}.jpg'.format(i))
                cv2.imwrite(path_to_file, img)
                i += 1
        else:
            print('Saving original image of {}'.format(image_name))
            for img in clip:
                # img = img.transpose(1,2,0)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # cv2.imshow('image',img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                path_to_file = os.path.join(path_to_dir, image_name + '_or_{}.jpg'.format(i))
                cv2.imwrite(path_to_file, img)
                i += 1

    clip_path = sorted(glob.glob(opts.input_path + r'/*'))  # get all the frames of a video
    clip_path = clip_path[opts.clip_s: opts.clip_s + opts.clip_length]  # pick a clip
    clip = []
    clip_image_ori = []
    clip_image_resize = []
    for frame_path in clip_path:
        img_arr = Image.open(frame_path)
        clip_image_ori.append(img_arr)
        img_arr = img_arr.resize((opts.input_size, opts.input_size))
        clip_image_resize.append(img_arr)
        img_arr = np.float32(img_arr)
        img_arr = normalize(img_arr)
        clip.append(img_arr)
    clip = np.array(clip)

    image_name = frame_path.split('/')[4]
    if opts.context:
        check_cropped_image(clip, image_name, crop_if=False)
    clip = clip.transpose(3, 0, 1, 2)
    clip_tensor = torch.from_numpy(clip).float()
    clip_tensor = clip_tensor.unsqueeze(0)
    clip_var = torch.autograd.Variable(clip_tensor, requires_grad=True)
    if opts.cuda:
        clip_var = clip_var.cuda()

    with open('center_mass_1f_otsu_filter_15.json') as handle:
        dict_center = json.loads(handle.read())

    crop_x, crop_y = calculate_center(dict_center[image_name], opts.clip_s)
    mp_crop_image = DestinedCrop(opts.input_size, crop_x, crop_y, image=True)
    mp_crop_arr = DestinedCrop(opts.input_size, crop_x, crop_y, image=False)

    clip_image_mp = [mp_crop_image(img) for img in clip_image_ori]
    clip_mp_arr = [[mp_crop_arr(img) for img in clip_image_ori]]

    if opts.context:
        check_cropped_image(clip_mp_arr[0], image_name, crop_if=True)
    clip_mp_np = np.array(clip_mp_arr)
    clip_mp_np = clip_mp_np.transpose(0, 4, 1, 2, 3)
    clip_mp_ten = torch.from_numpy(clip_mp_np).float()
    clip_mp_var = torch.autograd.Variable(clip_mp_ten, requires_grad=True)
    if opts.cuda:
        clip_mp_var = clip_mp_var.cuda()

    return clip_var, clip_mp_var, clip_image_resize, clip_image_mp
