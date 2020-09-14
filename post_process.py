import configparser
import os
import datetime
from user_screen import is_move
import pandas as pd


def add_one_day(start_date):
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = start_date + datetime.timedelta(days=1)
    return end_date.strftime("%Y-%m-%d")


def get_next_min_name(subject_name):
    split_list = subject_name.split('_')
    subject_date, subject_h, subject_m = split_list[-3:]
    if subject_m == '59':
        if subject_h == '23':
            subject_date = add_one_day(subject_date)
            subject_h = str(0).zfill(2)
        else:
            subject_h = str(int(subject_h) + 1).zfill(2)
        subject_m = str(0).zfill(2)
    else:
        subject_m = str(int(subject_m) + 1).zfill(2)
    split_list[-3:] = [subject_date, subject_h, subject_m]
    next_min_name = "_".join(split_list)
    return next_min_name


def get_next_2min_name(subject_name):
    split_list = subject_name.split('_')
    subject_date, subject_h, subject_m = split_list[-3:]
    subject_m = int(subject_m)
    if subject_m >= 58:
        if subject_h == '23':
            subject_date = add_one_day(subject_date)
            subject_h = str(0).zfill(2)
        else:
            subject_h = str(int(subject_h) + 1).zfill(2)
        subject_m = str((subject_m+2) % 60).zfill(2)
    else:
        subject_m = str(subject_m + 2).zfill(2)
    split_list[-3:] = [subject_date, subject_h, subject_m]
    next_min_name = "_".join(split_list)
    return next_min_name


def calculate_move(subject_name, source_dir):
    next_min_path = get_next_min_path(subject_name, source_dir)
    if os.path.exists(next_min_path):
        if is_move(next_min_path):
            return "Move"
        else:
            return "Not move"
    else:
        return "Not exist"


def post_process(predict_file, compare_csv, source_dir):
    f_w = open(compare_csv, "w")
    # detect name line is using \\ as separator or /
    separator = '\\' if '\\' in open(predict_file).readline() else '/'
    with open(predict_file) as f:
        while True:
            name_line = f.readline()
            subject_name = name_line.strip().split(separator)[-1].split('.')[0]
            result_line = f.readline()
            if not result_line:
                break
            gt_result = 1 if "1" in result_line else 0
            move_percent = calculate_move(subject_name, source_dir)
            f_w.write("{},{},{},{}\n".format(subject_name, 1, gt_result, move_percent))
            # f_w.write("{},{},{}\n".format(subject_name, 1, gt_result))


def preprocess_excel(csv_path, gt_path, dst_path):
    gt_dict = dict()
    with open(gt_path) as f:
        for line in f:
            hour, minute = [str(x).zfill(2) for x in line.strip().split(',')[:2]]
            if hour not in gt_dict.keys():
                gt_dict[hour] = [minute]
            else:
                gt_dict[hour].append(minute)
    f_w = open(dst_path, 'w')
    f_w.write('subject_period,predict_result,ground_truth,next_2min_move\n')
    data_frame = pd.read_csv(csv_path, header=None)
    data_frame.iloc[:, 0] = [x.split("\\")[-1].split('.')[0] for x in data_frame.iloc[:, 0]]
    data_frame.iloc[:, 1] = [int(x[1]) for x in data_frame.iloc[:, 1]]
    data_frame.iloc[:, 9] = [int(x.strip()[0]) for x in data_frame.iloc[:, 9]]
    # print(data_frame[1 in data_frame.iloc[:, 1:9]])
    all_move_name = data_frame.iloc[:, 0].tolist()
    for index, row in data_frame.iterrows():
        hour = row[0].strip().split('_')[-2]
        minute = row[0].strip().split('_')[-1]
        ground_truth = 0
        if hour in gt_dict.keys():
            if minute in gt_dict[hour]:
                ground_truth = 1
        next_min_name = get_next_min_name(row[0])
        next_2min_name = get_next_min_name(next_min_name)
        next_2min_move = 1 if (next_2min_name in all_move_name and next_min_name in all_move_name) else 0
        # if row[0] == "YJ_026_test_YJ_026_2020-07-23_YJ_026_2020-07-23_04_40":
        #     print("Pause")
        #     print("test")
        if 1 in row.iloc[1:10].values:
            f_w.write("{},{},{},{}\n".format(row[0], 1, ground_truth, next_2min_move))
        else:
            f_w.write("{},{},{},{}\n".format(row[0], 0, ground_truth, next_2min_move))


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    predict_file = config['Post Process']['predict_file']
    compare_csv = config['Post Process']['compare_csv']
    source_dir = config['Post Process']['source_dir']
    # epilepsy_file = config['Post Process']['epilepsy_file']
    post_process(predict_file, compare_csv, source_dir)


if __name__ == '__main__':
    preprocess_excel("/Users/jojen/Workspace/cityU/data/post_process/YJ_026_2020-07-23_predict.csv",
                     "/Users/jojen/Workspace/cityU/data/post_process/ground_truth.csv",
                     "/Users/jojen/Workspace/cityU/data/post_process/predict_result.csv")
