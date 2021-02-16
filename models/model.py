from __future__ import division
import torch
from torch import nn
from models import resnet
from models.resnet import get_fine_tuning_parameters
import os


def generate_model(args):
    model = resnet.resnet101(
            num_classes=args.classes,
            shortcut_type=args.resnet_shortcut,
            cardinality=args.resnet_cardinality,
            sample_size=args.sample_size,
            sample_duration=args.sample_duration,
            input_channels=args.input_channels,
            )
    if args.use_cuda:
        model = model.cuda()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = nn.DataParallel(model)

    return model, model.parameters()

