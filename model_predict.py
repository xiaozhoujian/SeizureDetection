from models.model import generate_model
from torch.utils.data import DataLoader
import torch
import os
import shutil
from torch import nn
import numpy as np
import linecache
from dataset.dataset_val_min import MiceOnline
import pickle


def make_list(test_name, output_dir, date, input_dir=None):
    """
    Make a list according to the intermediate directory under the day_dir
    :param test_name: string, the name of test, with the format of expert_subject_name_date.txt
    :param output_dir: string, the output dir which contains all the intermediate and result files
    :param date: string, the date of these specific videos
    :param input_dir: string, the directory of input data
    :return: string, the path to the video list file
    """
    print('Making list of {}'.format(test_name))
    file_list_path = os.path.join(output_dir, test_name)
    file = open(file_list_path, 'w')
    intermediate_dir = os.path.join(output_dir, 'intermediate', date)
    if input_dir:
        intermediate_dir = input_dir
    for video_file in os.listdir(intermediate_dir):
        line = os.path.join(intermediate_dir, video_file) + ' #0'
        file.write(line)
        file.write('\n')
    return file_list_path


def get_pretrained_model(args):
    """
    Load a pretrained model according to the arguments
    :param args: object, a config parser object containing the model-related information
    :return: DataParallel, a model with the trained weights and bias
    """
    print("Loading pretrained model...")
    cuda = True if torch.cuda.is_available() else False
    arch = '{}-{}'.format(args.model, args.model_depth)
    model, _ = generate_model(args)
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    split_path = os.path.split(args.resume_path)
    resume_path = os.path.join(cur_dir, *split_path)
    if resume_path:
        print('Loading checkpoint {}'.format(resume_path))
        if cuda:
            checkpoint = torch.load(resume_path)
        else:
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        assert arch == checkpoint['arch']
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if cuda:
        model = model.cuda()
    return model


def predict(args, model, file_list_path, output_dir, date,
            result_name="result", post_process=False, svm=False):
    test_data = MiceOnline(train=0, args=args, file_list_path=file_list_path)
    print('Preparing data loader...')
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=False)
    print("Length of test data loader is {}".format(len(test_data_loader)))

    result_path = os.path.join(output_dir, result_name, date)
    res_video_dir = os.path.join(result_path, 'res_videos')
    if not os.path.exists(res_video_dir):
        os.makedirs(res_video_dir)

    softmax = nn.Softmax(dim=1)
    sample_duration = args.sample_duration
    sample_size = args.sample_size

    with open('/data/jojen/classifier/RBF_SVM.pkl', 'rb') as fid:
        clf = pickle.load(fid)
    with torch.no_grad():
        # backup config.ini and new file object to record
        f_w = open(os.path.join(result_path, 'params.log'), 'w')
        f_w.write(str(args.__dict__))
        f_w.close()
        prob_log = open(os.path.join(result_path, 'prob.log'), 'w')
        index = -1
        for i, (clip, targets, video_name) in enumerate(test_data_loader):
            clip = torch.squeeze(clip)
            inputs = torch.Tensor(int(clip.shape[1] / sample_duration) + 1, 3, sample_duration, sample_size, sample_size)
            for k in range(inputs.shape[0] - 1):
                inputs[k, :, :, :, :] = clip[:, k * sample_duration:(k + 1) * sample_duration, :, :]

            inputs[-1, :, :, :, :] = clip[:, -sample_duration:, :, :]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            pre_label = torch.sum(outputs.topk(1)[1]).item()
            prob_outputs = softmax(outputs)
            prob = prob_outputs.cpu().data.numpy()

            index += 1
            if svm:
                if prob.shape == (10, 2):
                    pre_label = clf.predict((prob.reshape(1, 20)*100).astype(np.uint8))[0]
                else:
                    continue
                if pre_label == 1:
                    # in svm pre_label == 1 means it's case
                    prob_log.write('name: {}\nprob:\n{}\n'.format(video_name[0], (prob * 100).astype(np.uint8)))
                    prob_log.flush()
                    shutil.copy(linecache.getline(file_list_path, index + 1).strip()[:-3], res_video_dir)
            else:
                # Below is post process
                if post_process:
                    if pre_label >= 1:
                        for h in range(10):
                            if 0 < prob[h][1] - prob[h][0] < args.threshold:
                                # if prob of case bigger than control and their gap smaller than 0.3, it should be control
                                pre_label -= 1
                    if pre_label > 1:
                        prob_log.write('name: {}\nprob:\n{}\n'.format(video_name[0], (prob * 100).astype(np.uint8)))
                        prob_log.flush()
                        shutil.copy(linecache.getline(file_list_path, index + 1).strip()[:-3], res_video_dir)
                else:
                    if pre_label > 0:
                        prob_log.write('name: {}\nprob:\n{}\n'.format(video_name[0], (prob * 100).astype(np.uint8)))
                        prob_log.flush()
                        shutil.copy(linecache.getline(file_list_path, index + 1).strip()[:-3], res_video_dir)
