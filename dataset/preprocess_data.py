from __future__ import division
from PIL import Image
import numpy as np
import torch
import random
import numbers
import collections

try:
    import accimage
except ImportError:
    accimage = None

scale_choice = [1, 1 / 2 ** 0.25, 1 / 2 ** 0.5, 1 / 2 ** 0.75, 0.5]
crop_positions = ['c', 'tl', 'tr', 'bl', 'br']


def mpc(or_clip, re_clip, agent, steps_rl, targets, criterion):
    train_rl = False
    targets = torch.tensor([targets])
    targets = targets.cuda()
    state = agent.init_state(or_clip, re_clip, targets, criterion, train_rl)
    for steps in range(steps_rl):
        actions, _ = agent.select_action(state)
        state_ = agent.step(actions, or_clip, train_rl)
        state = state_
    agent.del_states()

    motion_patch = agent.tensor_crop(or_clip, actions)
    return motion_patch


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        """
        :param size: sequence or int, Desired output size.
            If size is a sequence like (w, h), output size self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
            self.crop_position = self. will be matched to this.
            If size is an int, smaller edge of the image will be matched to this number. i.e, if height > width,
            then image will be rescaled to (size * height / width, size)
        :param interpolation: image resize filter, default is PIL.Image.BILINEAR
        """
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        :param img: PIL.Image, image to be scaled.
        :return: Rescaled image
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
    """

    def __init__(self, size):
        """
        :param size: (sequence or int): Desired output size of the crop. If size is an int
        instead of sequence like (h, w), a square crop (size, size) is made.
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        :param img: Image to be cropped.
        :return: Cropped image
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        pass


class MultiScaleCornerCrop(object):
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    """

    def __init__(self, scale, size, crop_position, interpolation=Image.BILINEAR):
        """
        :param scale: float, cropping scales of the original size
        :param size: int, size of the smaller edge
        :param crop_position: list, list of position, options are  ['c', 'tl', 'tr', 'bl', 'br']
        :param interpolation: image resize filter, default is PIL.Image.BILINEAR
        """
        self.scale = scale
        self.size = size
        self.interpolation = interpolation
        self.crop_position = crop_position

    def __call__(self, img):

        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.crop_position == 'tr':
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height
        else:
            raise ValueError("Crop position {} is not support yet.".format(self.crop_position))

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)


class MultiScaleRandomCrop(object):
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    """

    def __init__(self, scale, size, crop_position, interpolation=Image.BILINEAR):
        """
        :param scale: float, cropping scales of the original size
        :param size: int, size of the smaller edge
        :param crop_position: list, list of position, options are  ['c', 'tl', 'tr', 'bl', 'br']
        :param interpolation: image resize filter, default is PIL.Image.BILINEAR
        """
        self.scale = scale
        self.size = size
        self.interpolation = interpolation
        self.crop_position = crop_position

    def __call__(self, img):
        # min_length = min(img.size[0], img.size[1])
        min_length = 224
        crop_size = int(min_length * self.scale)

        random_loc = random.random()

        x1 = int(32 * random_loc)
        y1 = int(32 * random_loc)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        """
        :param mean: sequence, sequence of means for R, G, B channels respectively
        :param std: sequence, sequence of standard deviations for R, G, B channels respectively
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Normalize given tensor image
        :param tensor: tensor, tensor image of size (C, H, W) to be normalized.
        :return: Normalized Image
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class RandomHorizontalFlip(object):
    """
    Horizontally flip the given PIL.Image randomly with a probability of 0.5.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


def get_mean(dataset='HMDB51'):
    """
    Get normalized mean values of different datasets
    :param dataset: string, choices options: ('activitynet', 'kinetics', 'HMDB51)
    :return: list of mean values
    """
    # assert dataset in ['activitynet', 'kinetics']
    if dataset == 'activitynet':
        return [114.7748, 107.7354, 99.4750]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [110.63666788, 103.16065604, 96.29023126]
    elif dataset == "HMDB51":
        return [0.36410178082273 * 255, 0.36032826208483 * 255, 0.31140866484224 * 255]
    else:
        raise ValueError("Mean value of dataset {} is not support".format(dataset))


def get_std(dataset='HMDB51'):
    """
    Get standard deviation of different datasets
    :param dataset: string, choices options: ('kinetics', 'HMDB51')
    :return: list of standard deviation values
    """
    # Kinetics (10 videos for each class)
    if dataset == 'kinetics':
        return [38.7568578, 37.88248729, 40.02898126]
    elif dataset == 'HMDB51':
        return [0.20658244577568 * 255, 0.20174469333003 * 255, 0.19790770088352 * 255]
    else:
        raise ValueError("Standard deviation value of dataset {} is not support".format(dataset))


def scale_crop(clip, dataset_type, opt):
    """Preprocess list(frames) based on train/test and modality.
    Training:
        - Multiscale corner crop
        - Random Horizonatal Flip (change direction of Flow accordingly)
        - Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor
        - Normalize R,G,B based on mean and std of ``ActivityNet``
    Testing/ Validation:
        - Scale frame
        - Center crop
        - Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor
        - Normalize R,G,B based on mean and std of ``ActivityNet``

    :param clip: list of RGB frames
    :param dataset_type: int, 1 for train, 0 for test
    :param opt: ArgumentParser, contains config options
    :return: Tensor(frames) of shape C X T X H X W
    """
    if opt.modality != 'RGB':
        raise ValueError("Modality {} is not support now.".format(opt.modality))
    processed_clip = torch.Tensor(3, len(clip), opt.sample_size, opt.sample_size)
    if dataset_type == 1:
        flip_prob = random.random()
        scale_factor = scale_choice[random.randint(0, len(scale_choice) - 1)]
        crop_position = crop_positions[random.randint(0, len(crop_positions) - 1)]
        for i, frame in enumerate(clip):
            frame = MultiScaleCornerCrop(scale=scale_factor, size=opt.sample_size, crop_position=crop_position)(frame)
            frame = RandomHorizontalFlip(p=flip_prob)(frame)
            frame = ToTensor(1)(frame)
            frame = Normalize(get_mean('kinetics'), [1, 1, 1])(frame)
            processed_clip[:, i, :, :] = frame
    else:
        for i, frame in enumerate(clip):
            frame = Scale(opt.sample_size)(frame)
            frame = CenterCrop(opt.sample_size)(frame)
            frame = ToTensor(1)(frame)
            frame = Normalize(get_mean('kinetics'), [1, 1, 1])(frame)
            processed_clip[:, i, :, :] = frame

    return processed_clip
