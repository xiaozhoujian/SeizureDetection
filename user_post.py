"""
This script is used to postprocess one-day videos prediction results
"""

import os
import shutil


def selection(pred_result, des_pos, des_neg):
    f = open(pred_result, 'r')
    lines = f.readlines()
    i = 0
    pos = 0
    neg = 0
    for line in lines:
        i += 1
        if i % 2 == 0:
            if str(1) in line:
                pos += 1
                shutil.copy(name.strip(), des_pos)
            else:
                neg += 1
                shutil.copy(name.strip(), des_neg)
        else:
            name = line
    f.close()
    return pos, neg


def main(day_dir):

    p_dir = os.path.join(day_dir, 'positive')
    n_dir = os.path.join(day_dir, 'negative')
    os.makedirs(p_dir)
    os.makedirs(n_dir)
    info = os.path.join(day_dir, 'info.txt')

    pos_n = 0
    neg_n = 0
    for f in os.listdir(day_dir):
        if f.endswith('_predict.txt'):
            i_file = os.path.join(day_dir, f)
            tmp_pos, tmp_neg = selection(i_file, p_dir, n_dir)
            pos_n += tmp_pos
            neg_n += tmp_neg
    total = pos_n + neg_n

    f = open(info, 'w')
    s = 'positive number: ' + str(pos_n) + '\n' + 'negative number: ' + str(neg_n) + '\n' + 'total number: ' + str(total) + '\n'
    f.write(s)
    f.close()
