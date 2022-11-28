import pandas as pd

from dataset.dataset_val_min import *
from dataset.preprocess_data import *

from torch.utils.data import DataLoader
import os
import torch
from torch import nn
from torch import optim
from opts import parse_opts
from models.model import generate_model
from torch.autograd import Variable
import time
from utils import AverageMeter, Logger, calculate_accuracy
import shutil


def output2dict(softmax, outputs, video_name, targets, is_train=False):
    prob_outputs = softmax(outputs)
    prob = prob_outputs.cpu().data.numpy()
    prob = prob * 100
    prob = np.round(prob.astype(float), 2)
    if is_train:
        res = []
        for i in range(len(video_name)):
            tmp_dict = {"video_name": video_name[i], "class": str(int(targets[i]))}
            for j in range(len(prob[i])):
                tmp_dict["prob{}".format(j)] = prob[i][j]
            res.append(tmp_dict)
        return res
    else:
        tmp_dict = {"video_name": video_name[0], "class": str(int(targets[0]))}
        x, y = prob.shape
        for x_idx in range(x):
            for y_idx in range(y):
                tmp_dict["prob{}{}".format(x_idx, y_idx)] = prob[x_idx][y_idx]
        return tmp_dict


def iterate_dataloader(dataloader, opts, model, softmax, epoch, val_file_name, val_logger=None, is_train=False):
    losses = AverageMeter()
    accuracies = AverageMeter()
    result_list = []
    if not is_train:
        fp_dir = os.path.join(opts.result_path, "predict_cases")
        if not os.path.exists(fp_dir):
            os.makedirs(fp_dir)
    with torch.no_grad():
        for i, (clip, targets, video_list) in enumerate(dataloader):
            clip = torch.squeeze(clip)
            if opts.modality == 'RGB':
                inputs = torch.Tensor(int(clip.shape[1] / opts.sample_duration) + 1, 3, opts.sample_duration,
                                      opts.sample_size, opts.sample_size)
            else:
                raise ValueError("Modality {} is not support yet.".format(opts.modality))

            for k in range(inputs.shape[0] - 1):
                inputs[k, :, :, :, :] = clip[:, k * opts.sample_duration:(k + 1) * opts.sample_duration, :, :]

            inputs[-1, :, :, :, :] = clip[:, -opts.sample_duration:, :, :]
            inputs = inputs.cuda()

            outputs = model(inputs)
            pre_label = torch.sum(outputs.topk(1)[1]).item()

            if targets.item() == 0 and pre_label == 0:
                acc = 1
            elif targets.item() == 1 and pre_label > 0:
                acc = 1
            else:
                acc = 0

            accuracies.update(acc, 1)

            if is_train:
                # validate after per epoch in training
                print("Video[{idx}]:\tpredict={pre_label}\ttrue={true_label}\tacc={acc}".format(
                    idx=i, pre_label=pre_label, true_label=targets[0], acc=accuracies.avg
                ))
            else:
                # if it is validating/test, save the false positive to further analyze
                print(video_list)
                video_name = video_list[0]
                dst = os.path.join(fp_dir, video_name)
                video_path = os.path.join(opts.val_path_1, video_name)
                shutil.copyfile(video_path, dst)
                prob_outputs = softmax(outputs)
                print("Video[{idx}]:\tname={video_name}\tpredict={pre_label}\ttrue={true_label}\tacc={acc}\noutput={probs}".format(
                    idx=i, video_name=video_name, pre_label=pre_label, true_label=targets[0],
                    acc=accuracies.avg, probs=prob_outputs
                ))
            # store predicted probabilities of every single video
            result_list.append(output2dict(softmax, outputs, video_list, targets, is_train=is_train))
    df = pd.DataFrame(result_list)
    if is_train:
        csv_file_name = '{}_epoch_{}.csv'.format(val_file_name[:-4], epoch)
        csv_path = os.path.join(opts.result_path, 'test_result', csv_file_name)
    else:
        csv_path = os.path.join(opts.result_path, 'probs.csv')
    df.to_csv(csv_path)
    if val_logger:
        if is_train:
            val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
        else:
            print(accuracies.avg)
            val_logger.log({'acc': accuracies.avg})


def initialization(opts, is_train=False):
    opts.arch = '{}-{}'.format(opts.model, opts.model_depth)
    torch.manual_seed(opts.manual_seed)

    log_path = os.path.join(opts.result_path, opts.dataset)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    print("Preprocessing validation data ...")
    val_data = globals()['{}'.format(opts.dataset)](dataset_type=2, opt=opts)
    print("Length of validation data = ", len(val_data))

    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=32, pin_memory=True,
                                drop_last=False)
    print("Length of validation dataloader = ", len(val_dataloader))

    # define the model
    print("Loading model... ", opts.model, opts.model_depth)
    model, parameters = generate_model(opts)

    if opts.resume_path1:
        begin_epoch = int(opts.resume_path1.split('.')[-2].split('_')[-1])
        overlay = False
    else:
        begin_epoch = 0
        overlay = True

    if opts.log == 1:
        val_logger_name = '{}_val_1_clip{}model{}{}.log'.format(opts.dataset, opts.sample_duration,
                                                                opts.model, opts.model_depth)
        headers = ['epoch', 'loss', 'acc'] if is_train else ['acc']
        val_logger = Logger(os.path.join(log_path, val_logger_name), headers, overlay=overlay)
    else:
        val_logger = None

    if is_train:
        print("Preprocessing train data ...")
        train_data = globals()['{}'.format(opts.dataset)](dataset_type=1, opt=opts)
        print("Length of train data = ", len(train_data))
        train_dataloader = DataLoader(train_data, batch_size=opts.batch_size, shuffle=True, num_workers=32,
                                      pin_memory=True, drop_last=True)
        print("Length of train dataloader = ", len(train_dataloader))
        criterion = nn.CrossEntropyLoss().cuda()
        print("Initializing the optimizer ...")
        if opts.nesterov:
            dampening = 0
        else:
            dampening = opts.dampening
        epoch_logger_name = '{}_train_clip{}model{}{}.log'.format(opts.dataset, opts.sample_duration,
                                                                  opts.model, opts.model_depth)
        epoch_logger = Logger(os.path.join(log_path, epoch_logger_name), ['epoch', 'loss', 'acc', 'lr'],
                              overlay=overlay)
        optimizer = optim.SGD(model.parameters(), lr=opts.learning_rate, momentum=opts.momentum, dampening=dampening,
                              weight_decay=opts.weight_decay, nesterov=opts.nesterov)

        if opts.resume_path1 != '':
            optimizer.load_state_dict(torch.load(opts.resume_path1)['optimizer'])
        print("lr = {} \t momentum = {} \t dampening = {} \t weight_decay = {}, \t nesterov = {}".format(
            opts.learning_rate, opts.momentum, dampening, opts.weight_decay, opts.nesterov))
        print("LR patience = ", opts.lr_patience)
        return begin_epoch, model, train_dataloader, criterion, optimizer, epoch_logger, val_data, val_dataloader, val_logger
    else:
        if opts.resume_path1:
            checkpoint = torch.load(opts.resume_path1, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model, val_data, val_dataloader, val_logger


def main():
    opts = parse_opts()
    print(opts)

    if opts.modality == 'RGB':
        opts.input_channels = 3
    else:
        raise ValueError("Modality {} is not support yet.".format(opts.modality))

    begin_epoch, model, train_dataloader, criterion, optimizer, epoch_logger, val_data, val_dataloader, val_logger\
        = initialization(opts, is_train=True)
    print('run')
    softmax = nn.Softmax(dim=1)
    min_loss = 1000
    min_epoch = 0

    log_path = os.path.join(opts.result_path, opts.dataset)
    overlay = False if opts.resume_path1 else True
    val_logger_name2 = '{}_val_2_clip{}model{}{}.log'.format(opts.dataset, opts.sample_duration,
                                                             opts.model, opts.model_depth)
    if opts.log == 1:
        val_logger_2 = Logger(os.path.join(log_path, val_logger_name2), ['epoch', 'loss', 'acc'], overlay=overlay)
    else:
        val_logger_2 = None

    csv_dir = os.path.join(opts.result_path, 'test_result')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    for epoch in range(begin_epoch, opts.n_epochs + 1):

        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        result_list = []
        for i, (inputs, targets, video_name) in enumerate(train_dataloader):
            data_time.update(time.time() - end_time)
            targets = targets.cuda(non_blocking=True)
            inputs = Variable(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
                  'Lr {lr}'.format(epoch, i + 1, len(train_dataloader), batch_time=batch_time, data_time=data_time,
                                   loss=losses, acc=accuracies, lr=optimizer.param_groups[-1]['lr']))
            # store predicted probabilities of every single video
            result_list.extend(output2dict(softmax, outputs, video_name, targets, is_train=True))
        df = pd.DataFrame(result_list)
        csv_file_name = '{}_epoch_{}.csv'.format(opts.train_file.strip().split('/')[-1][:-4], epoch)
        df.to_csv(os.path.join(csv_dir, csv_file_name))

        if opts.log == 1:
            epoch_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg,
                              'lr': optimizer.param_groups[-1]['lr']})

        if epoch % 5 == 0:
            # save epoch every 5 epoch
            save_file_path = os.path.join(opts.result_path,
                                          'save_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'arch': opts.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
        if losses.avg < min_loss:
            # save best.pth if loss less than previous minimum loss
            min_loss = losses.avg
            min_epoch = epoch
            save_file_path = os.path.join(opts.result_path,
                                          'best.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'arch': opts.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
        print("Current best epoch is {}, loss: {}".format(min_epoch, min_loss))

        model.eval()
        # val_file_1
        print("Length of validation data = ", len(val_data))
        iterate_dataloader(val_dataloader, opts, model, softmax, epoch, opts.val_file_1,
                           val_logger=val_logger, is_train=True)

        if opts.val_file_2:
            val_data_2 = globals()['{}'.format(opts.dataset)](dataset_type=3, opt=opts)
            val_dataloader_2 = DataLoader(val_data_2, batch_size=1,
                                          shuffle=False, num_workers=opts.n_workers, pin_memory=True, drop_last=False)
            print("Length of validation_2 data = ", len(val_data_2))
            iterate_dataloader(val_dataloader_2, opts, model, softmax, epoch, opts.val_file_2,
                               val_logger=val_logger_2, is_train=True)


if __name__ == "__main__":
    main()
