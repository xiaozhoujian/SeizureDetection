# -*- coding: utf-8 -*-
"""
author:john
Script to cut the video according to the excel(cut.xlsx)
"""
import os
import numpy as np
import shutil
import xlrd
import random
import configparser
import pandas as pd
import functools
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def cut_video(excel_path, new_dir, goal_paths):
    """
    Function used to cut the video according to the excel information
    :param excel_path: string, the path of the excel file
    :param new_dir: string, the path to storage the result
    :param goal_paths: dictionary, contains path of different classes
    """
    excel = xlrd.open_workbook(excel_path)
    table = excel.sheet_by_index(0)
    rows = table.nrows
    # First row is the video file name
    origin_video = table.col_values(0, start_rowx=0, end_rowx=None)
    start_time = [int(x) for x in table.col_values(1, start_rowx=0, end_rowx=None)]
    end_time = [int(x) for x in table.col_values(2, start_rowx=0, end_rowx=None)]
    # 4th row is the name of the classes
    type_new_ = [x.strip() for x in table.col_values(3, start_rowx=0, end_rowx=None)]
    for i in range(rows):
        filename, file_format = origin_video[i].split('.')
        new_name = os.path.join(new_dir, '{}_{}_{}.{}'.format(filename, str(start_time[i]).zfill(2),
                                                              str(end_time[i]).zfill(2), file_format))
        path = os.path.join(new_dir, origin_video[i])
        ffmpeg_extract_subclip(path, start_time[i], end_time[i], targetname=new_name)
        # all of the moving video will be copy to following path
        shutil.copy(new_name, goal_paths['move'])
        shutil.move(new_name, goal_paths[type_new_[i]])


def source2cut(source_excel, cut_excel, prefix):
    source_df = pd.read_excel(source_excel)
    filtered_df = source_df[functools.reduce(np.logical_and, [source_df['start_h'] == source_df['end_h'],
                                                              source_df['start_m'] == source_df['end_m'],
                                                              source_df['start_s'] < source_df['end_s']])]
    cut_list = []
    for _, row in filtered_df.iterrows():
        video_name = "{}_{}_{}.mp4".format(prefix, str(row['start_h']).zfill(2), str(row['start_m']).zfill(2))
        cut_list.append([video_name, row['start_s'], row['end_s'], row['classes']])
    df = pd.DataFrame(cut_list)
    df.to_excel(cut_excel, index=False, header=False)


# def main():
#     config = configparser.ConfigParser()
#     config.read('config.ini')
#     # all files in this directory
#     source_dir = config['Video Cutter']['dir']
#     cut_excel = config['Video Cutter']['cut_excel_path']
#     if config['Video Cutter'].getboolean('convert'):
#         source_excel = config['Video cutter']['source_excel_path']
#         prefix = "_".join(random.choice(os.listdir(source_dir)).split("_")[:-2])
#         source2cut(source_excel, cut_excel, prefix)
#
#     new_dir = source_dir
#     classes = ['move', 'grooming', 's-moving', 'standing_eating', 'walking', 'digging', 'misc_movement', 'epilepsy']
#     # all cut files will move to these directories
#     goal_paths = {x: os.path.join(new_dir, x) for x in classes}
#     for goal_path in goal_paths.values():
#         if not os.path.exists(goal_path):
#             os.makedirs(goal_path)
#
#     cut_video(cut_excel, new_dir, goal_paths)
#     print('Videos cut finished!')
#
#
# if __name__ == "__main__":
#     main()
