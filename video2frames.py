import cv2
import os
from PIL import Image
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import importlib.util
import argparse
import numpy as np
import sys
import time
import imageio
import re
from filter_noise import filter_directory
import configparser

sys.path.append("/home/jojen/workspace/RAFT/core")
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


def convert2of(video_path, model_path, dst_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()
    # args.small = True

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path))

    model = model.module
    model = model.to('cuda')
    model.eval()
    images = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if ret:
            # frame = cv2.resize(frame, (454, 256))
            images.append(torch.from_numpy(frame).permute(2, 0, 1).float())
        else:
            break
    print("Read frames finished")
    images = torch.stack(images, dim=0)
    padder = InputPadder(images.shape)
    images = padder.pad(images)[0]
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    # start_t = time.time()
    # print("Each prediction cost {}s".format(time.time() - start_t))
    # print("Prepare finished")
    out = imageio.get_writer(dst_path, fps=fps)
    with torch.no_grad():
        for i in range(images.shape[0] - 1):
            image1 = images[i, None].to('cuda')
            image2 = images[i + 1, None].to('cuda')

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            # out.write(viz(image1, flow_up).astype(np.uint8))
            out.append_data(viz(image1, flow_up).astype(np.uint8))
    # out.release()
    out.close()
    print("Time cost : {}s".format(time.time() - start_t))


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().detach().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().detach().numpy()
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    return img_flo[:, :, [2, 1, 0]]


def video2frames(video_path, dst_path):
    cap = cv2.VideoCapture(video_path)
    frame_cnt = 0
    frame_interval = 10
    interval = 0
    while frame_cnt < 10:
        _, frame = cap.read()
        interval += 1
        if interval > frame_interval:
            interval = 0
            frame = cv2.resize(frame, (1024, 436))
            im = Image.fromarray(frame)
            im.save(os.path.join(dst_path, "test_{}.png".format(frame_cnt)))
            frame_cnt += 1


def process_dir(directory, out_dir, model_path):
    videos = os.listdir(directory)
    for video in videos:
        video_path = os.path.join(directory, video)
        model_name = re.search(r'-(.*)\.', model_path).group(1)
        result_path = os.path.join(out_dir, "{}_{}".format(model_name, video))
        print(video_path)
        convert2of(video_path, model_path, result_path)


def main():
    # model_paths = ["/home/jojen/workspace/RAFT/models/raft-things.pth",
    #                "/home/jojen/workspace/RAFT/models/raft-sintel.pth"]
    model_paths = ["/home/jojen/workspace/RAFT/models/raft-sintel.pth"]
    config = configparser.ConfigParser()
    config.read('config.ini')
    of_dir = config['Optical Flow']['data_dir']
    data_dir = os.path.join(of_dir, 'source')
    optical_flow_dir = os.path.join(of_dir, 'optical_flow')
    if not os.path.exists(optical_flow_dir):
        os.makedirs(optical_flow_dir)
    print("Begin to generate optical flow")
    for model_path in model_paths:
        process_dir(data_dir, optical_flow_dir, model_path)
    filtered_dir = os.path.join(of_dir, 'filtered_dir')
    if not os.path.exists(filtered_dir):
        os.makedirs(filtered_dir)
    record_file = os.path.join(of_dir, 'result.csv')
    print('Begin to filter optical flow.')
    filter_directory(optical_flow_dir, filtered_dir, record_file)


def test_model_performance():
    model_dir = "/home/jojen/workspace/RAFT/models"
    models = os.listdir(model_dir)
    for model in models:
        model_path = os.path.join(model_dir, model)
        model_name = model.split('.')[0].split('-')[-1]
        convert2of("/data/jojen/026/026_case/case_YJ_026_2020-06-19_09_43.mp4",
                   model_path,
                   "/data/jojen/optical_flow/026/026_case/case_YJ_026_2020-06-19_09_43_{}.mp4".format(model_name))


if __name__ == '__main__':
    # video2frames("/Users/jojen/Workspace/cityU/data/test/test.mp4",
    #              "/Users/jojen/Workspace/cityU/data/test/test_raft")
    # test_model_performance()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()