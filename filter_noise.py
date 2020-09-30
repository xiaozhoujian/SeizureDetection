import cv2
import imageio
import numpy as np
import torch
import os
import time
import numpy
import sys
from torch.autograd import Variable

# dain_path = "/home/jojen/tools/DAIN"
# sys.path.append("/home/jojen/tools/DAIN")
# import networks


# def dain_interpolate(start_frame, end_frame, inter_frame_cnt):
#     if inter_frame_cnt == 0:
#         return None
#     else:
#         time_step = 2/(2+inter_frame_cnt)
#         # !python colab_interpolate.py --netName DAIN_slowmotion --time_step {fps/TARGET_FPS} --start_frame 1 --end_
#         # frame {pngs_generated_count} --frame_input_dir '{FRAME_INPUT_DIR}' --frame_output_dir '{FRAME_OUTPUT_DIR}'
#         model = networks.__dict__['DAIN_slowmotion'](timestep=time_step, training=False)
#
#         use_cuda = torch.cuda.is_available()
#         if use_cuda:
#             model = model.cuda()
#
#         model_path = os.path.join(dain_path, 'model_weights/best.pth')
#         if not os.path.exists(model_path):
#             print("*****************************************************************")
#             print("**** We couldn't load any trained weights ***********************")
#             print("*****************************************************************")
#             exit(1)
#
#         if use_cuda:
#             pretrained_dict = torch.load(model_path)
#         else:
#             pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
#
#         model_dict = model.state_dict()
#         # 1. filter out unnecessary keys
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#         # 2. overwrite entries in the existing state dict
#         model_dict.update(pretrained_dict)
#         # 3. load the new state dict
#         model.load_state_dict(model_dict)
#         # 4. release the pretrained dict for saving memory
#         model = model.eval()  # deploy mode
#
#         # len of time offsets is the interval frames number between input frame and end frame
#         time_offsets = [kk * time_step for kk in range(1, int(1.0 / time_step))]
#
#         torch.set_grad_enabled(False)
#
#         # we want to have input_frame between (start_frame-1) and (end_frame-2)
#         # this is because at each step we read (frame) and (frame+1)
#         # so the last iteration will actuall be (end_frame-1) and (end_frame)
#         if use_cuda:
#             start_frame = torch.from_numpy(np.transpose(start_frame, (2, 0, 1)).astype("float32") /
#                                            255.0).type(torch.cuda.FloatTensor)
#             end_frame = torch.from_numpy(np.transpose(end_frame, (2, 0, 1)).astype("float32") /
#                                          255.0).type(torch.cuda.FloatTensor)
#         else:
#             start_frame = torch.from_numpy(np.transpose(start_frame, (2, 0, 1)).astype("float32") /
#                                            255.0).type(torch.FloatTensor)
#             end_frame = torch.from_numpy(np.transpose(end_frame, (2, 0, 1)).astype("float32") /
#                                          255.0).type(torch.FloatTensor)
#
#         width = start_frame.size(2)
#         height = start_frame.size(1)
#
#         # if int width is the multiple of 2**7(128)
#         if width != ((width >> 7) << 7):
#             # equal to intwidth multiple by 2
#             width_pad = (((width >> 7) + 1) << 7)  # more than necessary
#             padding_left = int((width_pad - width) / 2)
#             padding_right = width_pad - width - padding_left
#         else:
#             padding_left = 32
#             padding_right = 32
#
#         if height != ((height >> 7) << 7):
#             height_pad = (((height >> 7) + 1) << 7)  # more than necessary
#             padding_top = int((height_pad - height) / 2)
#             padding_bottom = height_pad - height - padding_top
#         else:
#             padding_top = 32
#             padding_bottom = 32
#
#         pader = torch.nn.ReplicationPad2d([padding_left, padding_right, padding_top, padding_bottom])
#
#         start_frame = pader(Variable(torch.unsqueeze(start_frame, 0)))
#         end_frame = pader(Variable(torch.unsqueeze(end_frame, 0)))
#
#         if use_cuda:
#             start_frame = start_frame.cuda()
#             end_frame = end_frame.cuda()
#
#         #  cur_outputs,cur_offset_output,cur_filter_output
#         # offset is list with len 2, each of its element with shape [1, 2, height, width]
#         # filter is list with len 2, each of its element with shape [1, 16, height, width]
#         y_s, offset, filter = model(torch.stack((start_frame, end_frame), dim=0))
#         # y_s is list, with shape [2, len(time_offsets)], each of it with shape [1, 3 height, width]
#         y_ = y_s[1]
#
#         if use_cuda:
#             if not isinstance(y_, list):
#                 y_ = y_.data.cpu().numpy()
#             else:
#                 y_ = [item.data.cpu().numpy() for item in y_]
#         else:
#             if not isinstance(y_, list):
#                 y_ = y_.data.numpy()
#             else:
#                 y_ = [item.data.numpy() for item in y_]
#         # after transpose y_[0] become [height, width, 3]
#         y_ = [np.transpose(255.0 * item.clip(0, 1.0)[0, :, padding_top:padding_top + height,
#                                    padding_left:padding_left + width], (1, 2, 0)) for item in y_]
#
#         interpolated_frame_number = 0
#         for item, time_offset in zip(y_, time_offsets):
#             interpolated_frame_number += 1
#             output_frame_file_path = os.path.join("result", "{}.png".format(time_offset))
#             imageio.imsave(output_frame_file_path, np.round(item).astype(numpy.uint8))
#         # return np.round(item).astype(numpy.uint8)

# def softmax_interporlate(begin_frame, end_frame, inter_frame_cnt):


def is_noise(frame):
    """Resize and convert to gray"""
    frame = frame[int(frame.shape[0]/2):, :, :]
    frame = cv2.resize(frame, (455, 256))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # half_height = int(frame.shape[0] / 2)
    white_pixel_cnt = np.sum((gray_frame.reshape(gray_frame.shape[0]*gray_frame.shape[1]) > 250))
    total_pixel_cnt = gray_frame.shape[0] * gray_frame.shape[1]
    if white_pixel_cnt/total_pixel_cnt >= 0.6:
        return False
    else:
        return True


def filter_noise(video_path, dst_path):
    # cap = cv2.VideoCapture(video_path)
    print('Processing {}'.format(video_path))
    reader = imageio.get_reader(video_path)
    out = imageio.get_writer(dst_path, fps=reader.get_meta_data()['fps'])
    source_frame_cnt = reader.get_meta_data()['fps'] * reader.get_meta_data()['duration']
    filtered_frame_cnt = 0
    # frame2 = None
    for _, im in enumerate(reader):
        if not is_noise(im):
            # frame1, frame2 = frame2, im
            # if frame1 and frame2:
            #     frames = dain_interpolate(frame1, frame2, noise_cnt)
            #     [out.append_data(x) for x in frames]
            out.append_data(im)
            filtered_frame_cnt += 1
    return source_frame_cnt, filtered_frame_cnt


def filter_directory(directory, output_dir, record_file):
    video_files = os.listdir(directory)
    record = open(record_file, 'w')
    record.write("source_path,output_path,source_frames,filtered_frames,clean%\n")
    for video_file in video_files:
        video_path = os.path.join(directory, video_file)
        output_path = os.path.join(output_dir, video_file)
        source_fps, filtered_fps = filter_noise(video_path, output_path)
        record.write("{},{},{},{},{}\n".format(video_path, output_path, source_fps, filtered_fps, filtered_fps/source_fps))


# def test_dain():
#     start_frame_path = ""
#     end_frame_path = ""
#     start_frame = imageio.imread(start_frame_path)
#     end_frame = imageio.imread(end_frame_path)
#     # dain_interpolate(start_frame, end_frame, inter_frame_cnt=2)


def main():
    # filter_noise("/home/jojen/optical_flow/video_raft_kitti.mp4",
    #              "/home/jojen/optical_flow/result.mp4")
    directory = '/data/jojen/optical_flow/test/noise_video'
    output_dir = '/data/jojen/optical_flow/test/filtered_video'
    record_file = '/data/jojen/optical_flow/test/record.csv'
    filter_directory(directory, output_dir, record_file)


if __name__ == '__main__':
    main()
