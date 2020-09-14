import numpy as np
import imutils
import cv2
from models.model import generate_model
import torch
import configparser
import decord


def get_pretrained_model():
    config = configparser.ConfigParser()
    config.read('config.ini')
    print("[INFO] loading human activity recognition model...")
    cuda = True if torch.cuda.is_available() else False
    arch = '{}-{}'.format(config.get('Network', 'model'), config.get('Network', 'model_depth'))
    model, _ = generate_model(config)
    resume_path1 = config.get('Network', 'resume_path1')
    if resume_path1:
        print('loading checkpoint {}'.format(resume_path1))
        if cuda:
            checkpoint = torch.load(resume_path1)
        else:
            checkpoint = torch.load(resume_path1, map_location=torch.device('cpu'))
        assert arch == checkpoint['arch']
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def main(model, sample_size, sample_duration, inputs):
    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture(inputs)
    # f = open(result_path, 'a')
    # f.write(inputs.split(separator)[-1])
    labels_list = []
    while True:
        # initialize the batch of frames that will be passed through the model
        frames = []
        # loop over the number of required sample frames
        for i in range(sample_duration):
            grabbed, frame = vs.read()

            # if the frame was not grabbed then we've reached the end of the video stream so exit the script
            if not grabbed:
                print("Predict result of {} is {}".format(inputs, labels_list))
                return inputs, labels_list
            # resize the height to 112 pixels, width is also changed, it's down sample
            frame = imutils.resize(frame, height=112)
            frames.append(frame)

        # now that our frames array is filled we can construct our blob
        blob = cv2.dnn.blobFromImages(frames, 1.0, size=(sample_size, sample_size), mean=(114.7748, 107.7354, 99.4750),
                                      swapRB=True, crop=False)

        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)

        # pass the blob through the network to obtain our human activity recognition predictions
        model_inputs = torch.from_numpy(blob)
        outputs = model(model_inputs)
        label = np.array(torch.mean(outputs, dim=0, keepdim=True).topk(1)[1].cpu().data[0])
        labels_list.append(label[0])
        # f.write(",{}".format(label[0]))


# if __name__ == '__main__':
#     inputs = "/Volumes/WD_BLACK/test_P_remove_YJ_026_20200627-20200712/YJ_026_20200627-20200712_Case/YJ_026_2020-06-27_18_28.mp4"
#     result_path = "/Users/jojen/Workspace/cityU/data/predict_result.txt"
#     main(inputs, result_path)
