# -*- coding: utf-8 -*-
"""
author:john
Script to cut the vedio according to the excel(cut.xlsx)
"""
import numpy as np
import os
import shutil
import xlrd
import cv2
import config
import random
import time



def move_case(excel_path, newdir, goal_path1, goal_path2, goal_path3, goal_path4, goal_path5, goal_path6):
    global new_name
    time_start = time.time()
    excel = xlrd.open_workbook(excel_path)
    table = excel.sheet_by_index(0)
    rows = table.nrows
    # 表格第一列为视频名名，第二列为开始时间，第三列结束时间，第四列类型 1 case 2 control
    ori_video = table.col_values(0, start_rowx=0, end_rowx=None)
    start_time = table.col_values(1, start_rowx=0, end_rowx=None)
    end_time = table.col_values(2, start_rowx=0, end_rowx=None)
    type_ = table.col_values(3, start_rowx=0, end_rowx=None)
    type_new_ = table.col_values(4, start_rowx=0, end_rowx=None)

    for i in range(rows):
        if type_[i] == 1:
            new_name = 'class_' + ori_video[i]
            path = os.path.join(newdir, ori_video[i])
            duration = end_time[i] - start_time[i]
            os.system('ffmpeg -y -i %s -ss %i -t %i -codec copy %s' % (path, start_time[i], duration, new_name))
            shutil.copy(new_name, goal_path1)
        if type_[i] == 2:
            new_name = 'class2_' + ori_video[i]
            path = os.path.join(newdir, ori_video[i])
            duration = end_time[i] - start_time[i]
            os.system('ffmpeg -y -i %s -ss %i -t %i -codec copy %s' % (path, start_time[i], duration, new_name))
            shutil.copy(new_name, goal_path1)
        if type_[i] == 3:
            new_name = 'class3_' + ori_video[i]
            path = os.path.join(newdir, ori_video[i])
            duration = end_time[i] - start_time[i]
            os.system('ffmpeg -y -i %s -ss %i -t %i -codec copy %s' % (path, start_time[i], duration, new_name))
            shutil.copy(new_name, goal_path1)
        if type_new_[i] == 's-moving':
            path = os.path.join(newdir, ori_video[i])
            duration = end_time[i] - start_time[i]
            os.system('ffmpeg -y -i %s -ss %i -t %i -codec copy %s' % (path, start_time[i], duration, new_name))
            shutil.move(new_name, goal_path2)
        if type_new_[i] == 'standing/eating':
            path = os.path.join(newdir, ori_video[i])
            duration = end_time[i] - start_time[i]
            os.system('ffmpeg -y -i %s -ss %i -t %i -codec copy %s' % (path, start_time[i], duration, new_name))
            shutil.move(new_name, goal_path3)
        if type_new_[i] == 'walking':
            path = os.path.join(newdir, ori_video[i])
            duration = end_time[i] - start_time[i]
            os.system('ffmpeg -y -i %s -ss %i -t %i -codec copy %s' % (path, start_time[i], duration, new_name))
            shutil.move(new_name, goal_path4)
        if type_new_[i] == 'digging':
            path = os.path.join(newdir, ori_video[i])
            duration = end_time[i] - start_time[i]
            os.system('ffmpeg -y -i %s -ss %i -t %i -codec copy %s' % (path, start_time[i], duration, new_name))
            shutil.move(new_name, goal_path5)
        if type_new_[i] == 'misc movement':
            path = os.path.join(newdir, ori_video[i])
            duration = end_time[i] - start_time[i]
            os.system('ffmpeg -y -i %s -ss %i -t %i -codec copy %s' % (path, start_time[i], duration, new_name))
            shutil.move(new_name, goal_path6)


def main():
    time_start = time.time()
    dir = 'E:\\epilepsy_video\\026\\2020-07-15\\move'  # all files in this directionary
    newdir = dir  # all files are moved in this directionary
    excel_path = 'cut.xlsx'  # excel_path
    goal_path1 = 'E:\\epilepsy_video\\026\\2020-07-15\\move_1'  # case files are moved in this direcitionary
    if not os.path.exists(goal_path1):
        os.makedirs(goal_path1)
    goal_path2 = 'E:\\epilepsy_video\\026\\2020-07-15\\s-moving'  # control files are moved in this direcitionary
    if not os.path.exists(goal_path2):
        os.makedirs(goal_path2)
    goal_path3 = 'E:\\epilepsy_video\\026\\2020-07-15\\standing_eating'  # preprocessed case files are moved in this direcitionary
    if not os.path.exists(goal_path3):
        os.makedirs(goal_path3)
    goal_path4 = 'E:\\epilepsy_video\\026\\2020-07-15\\walking'  # preprocessed control files are moved in this direcitionary
    if not os.path.exists(goal_path4):
        os.makedirs(goal_path4)
    goal_path5 = 'E:\\epilepsy_video\\026\\2020-07-15\\digging'  # preprocessed control files are moved in this direcitionary
    if not os.path.exists(goal_path5):
        os.makedirs(goal_path5)
    goal_path6 = 'E:\\epilepsy_video\\026\\2020-07-15\\misc_movement'  # preprocessed control files are moved in this direcitionary
    if not os.path.exists(goal_path6):
        os.makedirs(goal_path6)
    move_case(excel_path, newdir, goal_path1, goal_path2, goal_path3, goal_path4, goal_path5, goal_path6)
    print('finish move_case')


if __name__ == "__main__":
    main()
