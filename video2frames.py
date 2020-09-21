import cv2
import os
from PIL import Image
import torch
import importlib.util
import argparse
import numpy as np
import sys
import time
import imageio

sys.path.append("/Users/jojen/Workspace/cityU/optical_flow/RAFT/core")
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


def convert2of(video_path, model_path, dst_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model = model.module
    model.to("cpu")
    model.eval()
    images = []
    cap = cv2.VideoCapture(video_path)
    used_count = 0
    while True:
        ret, frame = cap.read()
        if used_count < 50:
            images.append(torch.from_numpy(frame).permute(2, 0, 1).float())
            used_count += 1
        else:
            break
    print("Read frames finished")
    images = torch.stack(images, dim=0)
    images = images.to("cpu")
    padder = InputPadder(images.shape)
    images = padder.pad(images)[0]
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)

    image1 = images[0, None]
    image2 = images[1, None]
    start_t = time.time()
    with torch.no_grad():
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    print("Each prediction cost {}s".format(time.time() - start_t))
    output_image = viz(image1, flow_up)
    # cv2.imshow('', output_image)
    # cv2.waitKey()
    print("Prepare finished")
    out = cv2.VideoWriter(dst_path, fourcc, fps, (output_image.shape[0], output_image.shape[1]))
    with torch.no_grad():
        for i in range(images.shape[0] - 1):
            image1 = images[i, None]
            image2 = images[i + 1, None]

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            out.write(viz(image1, flow_up))
    out.release()


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().detach().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().detach().numpy()
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    return img_flo[:, :, [2, 1, 0]] / 255.0


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


if __name__ == '__main__':
    # video2frames("/Users/jojen/Workspace/cityU/data/test/test.mp4",
    #              "/Users/jojen/Workspace/cityU/data/test/test_raft")
    convert2of("/Users/jojen/Workspace/cityU/data/test/test.mp4",
               "/Users/jojen/Workspace/cityU/optical_flow/RAFT/models/raft-things.pth",
               "/Users/jojen/Workspace/cityU/data/test/test_of.mp4")