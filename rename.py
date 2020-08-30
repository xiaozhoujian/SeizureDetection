import os
import time
def rename(input):
 elems = os.listdir(input)
 for elem in elems:
     name = elem[14:-4]
     path = os.path.join(input, elem)
     os.rename(path, input + 'control_YJ_L04_2020-' + name +'.mp4')

input = 'E:\\dataset_4\\rename\\'
rename(input)