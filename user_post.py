"""
This script is used to postprocess one-day videos prediction results
"""

import getopt
import os, sys
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

opts, args = getopt.getopt(sys.argv[1:], '-h-d:')
day_dir = ''
for opt_name, opt_value in opts:
    if opt_name == '-d':
        day_dir = opt_value

if day_dir == '' or (not os.path.exists(day_dir)):
    print('prediction result does not exist.')

p_dir = os.path.join(day_dir, 'positive')
n_dir = os.path.join(day_dir, 'negative')
os.makedirs(p_dir)
os.makedirs(n_dir)
infor = os.path.join(day_dir, 'info.txt')

pos_n = 0
neg_n = 0
total = 0
for f in os.listdir(day_dir):
    if f.endswith('_predict.txt'):
        i_file = os.path.join(day_dir, f)
        tmp_pos, tmp_neg = selection(i_file, p_dir, n_dir)
        pos_n += tmp_pos
        neg_n += tmp_neg
total = pos_n + neg_n

f = open(infor, 'w')
s = 'positive number: ' + str(pos_n) + '\n' + 'negative number: ' + str(neg_n) + '\n' + 'total number: ' + str(total) + '\n'
f.write(s)
f.close()







