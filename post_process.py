import configparser
import os
import datetime
from user_screen import is_move


def add_one_day(start_date):
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = start_date + datetime.timedelta(days=1)
    return end_date.strftime("%Y-%m-%d")


def get_next_min_path(subject_name, source_dir):
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
    next_min_file = "_".join(split_list) + ".mp4"
    next_min_path = os.path.join(source_dir, next_min_file)
    return next_min_path


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


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    predict_file = config['Post Process']['predict_file']
    compare_csv = config['Post Process']['compare_csv']
    source_dir = config['Post Process']['source_dir']
    # epilepsy_file = config['Post Process']['epilepsy_file']
    post_process(predict_file, compare_csv, source_dir)


if __name__ == '__main__':
    main()
