import cv2
import imageio
import numpy as np


def filter_noise(video_path, dst_path):
    cap = cv2.VideoCapture(video_path)
    out = imageio.get_writer(dst_path, fps=cap.get(cv2.CAP_PROP_FPS))
    while True:
        ret, frame = cap.read()
        white_cnt = 0
        # tmp_array = frame[half_height:, :]
        if ret:
            half_height = int(frame.shape[0] / 2)
            for i in range(half_height, frame.shape[0]):
                for j in range(frame.shape[1]):
                    if np.sum(frame[i, j] - [250, 250, 250]) > 0:
                        white_cnt += 1
        else:
            break
        print(white_cnt / (half_height * frame.shape[1]))
        cv2.imshow('', frame)
        cv2.waitKey(5)
        if white_cnt/(half_height*frame.shape[1]) >= 0.6:
            out.append_data(frame)


def main():
    filter_noise("/Users/jojen/Workspace/cityU/data/move/video_raft_kitti.mp4",
                 "/Users/jojen/Workspace/cityU/data/move/result.mp4")


if __name__ == '__main__':
    main()
