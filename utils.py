from __future__ import division
import ntpath
from dateutil.parser import parse


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def is_date(string, fuzzy=False):
    """
    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    :return whether the string can be interpreted as a date.
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False
