'''
this is for user to use the model and predict
input should be a directory of day
'''

import getopt
import os, sys
import time


pj_dir = os.getcwd()
rename_extract_exe = os.path.join(pj_dir, 'user_screen.py')
preprocess_exe = os.path.join(pj_dir, 'user_preprocess.py')
predict_exe = os.path.join(pj_dir, 'user_predict.py')
merge_exe = os.path.join(pj_dir, 'user_merge.py')
extract_exe = os.path.join(pj_dir, 'user_extraction.py')
post_exe = os.path.join(pj_dir, 'user_post.py')
annotation_path = os.path.join(pj_dir, 'mice_labels', 'class.txt')
model_load = os.path.join(pj_dir, 'results_mice_resnext101', 'model_mice_0723.pth')

opts, args = getopt.getopt(sys.argv[1:], '-h-d:-e:-n:')
day_dir = ''
num = ''
exper = ''
pj_dir = os.getcwd()
for opt_name, opt_value in opts:
    if opt_name == '-d':
        day_dir = opt_value
    if opt_name == '-e':
        exper = opt_value
    if opt_name == '-n':
        num = opt_value

if day_dir == '':
    print('Please input day video directory.')
    sys.exit()


st = time.time()
print('start screening.........')
rename_extract_cmd = 'python ' + rename_extract_exe + ' -d ' + day_dir + ' -e ' + exper + ' -n ' + num
os.system(rename_extract_cmd)

et = time.time()
used_t = (et-st)/60
print('Prediction compeleted')
print('Used time :' + str(used_t) + 'min')
