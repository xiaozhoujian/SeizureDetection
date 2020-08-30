import getopt
import os, sys
import time


pj_dir = os.getcwd()
day_exe = os.path.join(pj_dir, 'user_drive.py')

opts, args = getopt.getopt(sys.argv[1:], '-h-i:-e:-n:')
outer_dir = ''
num = ''
exper = ''
for opt_name, opt_value in opts:
    if opt_name == '-i':
        outer_dir = opt_value
    if opt_name == '-e':
        exper = opt_value
    if opt_name == '-n':
        num = opt_value

if outer_dir == '':
    print('Please input outer video directory.')
    sys.exit()

day_list = os.listdir(outer_dir)
for d in day_list:
    day_dir = os.path.join(outer_dir, d)
    cmd = 'python ' + day_exe + ' -d ' + day_dir + ' -e ' + exper + ' -n ' + num
    os.system(cmd)