from models.model import generate_model
from torch.utils.data import DataLoader
import torch
import os
import shutil
from utils import AverageMeter
from torch import nn
import numpy as np
import linecache
from dataset.dataset_val_min import MiceOnline


def make_list(test_name, day_dir):
    """
    Make a list according to the intermediate directory under the day_dir
    :param test_name: string, the name of test, with the format of expert_subject_name_date.txt
    :param day_dir: string, the day dir which contains all the videos
    :return: string, the path to the video list file
    """
    print('Making list of {}'.format(test_name))
    file_list_path = os.path.join(day_dir, test_name)
    file = open(file_list_path, 'w')
    intermediate_dir = os.path.join(day_dir, 'intermediate')
    for video_file in os.listdir(intermediate_dir):
        line = os.path.join(intermediate_dir, video_file) + ' #0'
        file.write(line)
        file.write('\n')
    return file_list_path


def get_pretrained_model(config):
    """
    Load a pretrained model according to the config.ini file
    :param config: object, a config parser object containing the model-related information
    :return: DataParallel, a model with the trained weights and bias
    """
    print("Loading pretrained model...")
    cuda = True if torch.cuda.is_available() else False
    arch = '{}-{}'.format(config.get('Network', 'model'), config.get('Network', 'model_depth'))
    model, _ = generate_model(config)
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    resume_path = os.path.join(cur_dir, config.get('Path', 'resume_path'))
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


def predict(config, model, file_list_path, pj_dir):
    dataset = config.get('Network', 'dataset')
    test_data = MiceOnline(train=0, config=config, file_list_path=file_list_path)
    print('Preparing data loader...')
    test_data_loader = DataLoader(test_data, batch_size=config.getint('Network', 'batch_size'),
                                  shuffle=False, num_workers=config.getint('Network', 'num_workers'),
                                  pin_memory=True, drop_last=False)
    print("Length of test data loader is {}".format(len(test_data_loader)))

    accuracies = AverageMeter()
    result_path = os.path.join(config.get('Path', 'source_dir'), 'result_{}'.format(dataset))
    res_video_dir = os.path.join(result_path, 'res_videos')
    if not os.path.exists(res_video_dir):
        os.makedirs(res_video_dir)

    softmax = nn.Softmax(dim=1)
    sample_duration = config.getint('Network', 'sample_duration')
    sample_size = config.getint('Network', 'sample_size')

    with torch.no_grad():
        # backup config.ini and new file object to record
        shutil.copyfile(os.path.join(pj_dir, 'config.ini'), os.path.join(result_path, 'config.ini'))
        f_w = open(os.path.join(result_path, 'params.log'), 'w')
        prob_log = open(os.path.join(result_path, 'prob.log'), 'w')
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
            # if pre_label > 1:
            #     import ipdb
            #     ipdb.set_trace()
            prob_outputs = softmax(outputs)
            prob = prob_outputs.cpu().data.numpy()
            pre_control = []
            index = -1
            # Belong is post process
            if targets.item() == 0:
                index += 1
                pre_control.append(pre_label)

                if pre_label > 1:
                    for h in range(10):
                        if 0 < prob[h][1] - prob[h][0] < 0.3:
                            # if prob of case bigger than control and their gap smaller than 0.3, it should be control
                            pre_label = pre_label - 1
                    if pre_label > 1:
                        # acc = 0 means it's a case
                        acc = 0
                        prob_log.write('name: {}\nprob:\n{}\n'.format(video_name[0], (prob * 100).astype(np.uint8)))
                        prob_log.flush()

                        shutil.copy(linecache.getline(file_list_path, index + 1).strip()[:-3], res_video_dir)
                    else:
                        acc = 1
                elif (pre_label > 5 and pre_control[index - 1] > 2) or pre_label == 0:
                    # if current have 5+ case frames and previous predict label bigger 2
                    acc = 1

            accuracies.update(acc, inputs.size(0))

            f_w.write("Video[{}]:\t predict = {}\t true = {}\t acc = {}\n".format(i, pre_label,
                                                                                  targets[0], accuracies.avg))
            f_w.flush()
        f_w.write("Video accuracy = " + str(accuracies.avg) + '\n')
