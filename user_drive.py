'''
this is for user to use the model and predict
input should be a directory of day
'''

import getopt
import os
import sys
import time


pj_dir = os.getcwd()
rename_extract_exe = os.path.join(pj_dir, 'user_screen.py')
preprocess_exe = os.path.join(pj_dir, 'user_preprocess.py')
predict_exe = os.path.join(pj_dir, 'user_predict.py')
merge_exe = os.path.join(pj_dir, 'user_merge.py')
extract_exe = os.path.join(pj_dir, 'user_extraction.py')
post_exe = os.path.join(pj_dir, 'user_post.py')
annotation_path = os.path.join(pj_dir, 'mice_labels', 'class.txt')
model_load = os.path.join(pj_dir, 'results_mice_resnext101', 'model_mice_0715.pth')

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
print('start predicting.........')
rename_extract_cmd = 'python ' + rename_extract_exe + ' -d ' + day_dir + ' -e ' + exper + ' -n ' + num
os.system(rename_extract_cmd)

hour_list = os.listdir(day_dir)
hour_list.sort()

preprocess_cmd = 'python ' + preprocess_exe + ' -d ' + day_dir
print(preprocess_cmd)
os.system(preprocess_cmd)

for h_dir in hour_list:
    predict_dir = os.path.join(day_dir, h_dir)
    print('Predicting videos in ' + predict_dir)
    save_path = os.path.join(day_dir, h_dir + '_predict.txt')
    video_list = os.listdir(predict_dir)
    video_list.sort()
    for file in video_list:
        predict_cmd = 'python user_predict.py --n_classes 2 ' \
                      '--model resnext --model_depth 101 --sample_duration 64 ' \
                      '--annotation_path ' + annotation_path  + \
                      ' --resume_path1 ' + model_load +\
                      ' --inputs ' + os.path.join(predict_dir, file) + \
                      ' --result_path ' + save_path
        #print(predict_cmd)
        os.system(predict_cmd)
    print('-------------------------------------------------------------')
    print()


post_cmd = 'python ' + post_exe + ' -d ' + day_dir
print(post_cmd)
os.system(post_cmd)

et = time.time()
used_t = (et-st)/60
print('Prediction compeleted')
print('Used time :' + str(used_t) + 'min')
