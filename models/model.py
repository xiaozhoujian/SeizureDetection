from __future__ import division
import torch
from torch import nn
from models import resnext, resnet2p1d, resnet, s3dg
from models.resnext import get_fine_tuning_parameters


def generate_model(opt):
    if opt.model == 'resnext':
        if opt.model_depth == 101:
            model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                input_channels=opt.input_channels,
                output_layers=opt.output_layers)
        elif opt.model_depth == 152:
            model = resnext.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                input_channels=opt.input_channels,
                output_layers=opt.output_layers)
        elif opt.model_depth == 50:
            model = resnext.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                input_channels=opt.input_channels,
                output_layers=opt.output_layers)
        elif opt.model_depth == 18:
            model = resnext.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                input_channels=opt.input_channels,
                output_layers=opt.output_layers)
        else:
            raise "Model Depth Error: Depth not 18, 50 or 101!"

        model = model.cuda()
        model = nn.DataParallel(model)
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)

            # assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])
            model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
            model.module.fc = model.module.fc.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters
        elif opt.resume_path1:
            print("Reuming pretrained model {}".format(opt.resume_path1))
            pretrain = torch.load(opt.resume_path1)
            assert opt.arch == pretrain['arch']
            model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
            model.module.fc = model.module.fc.cuda()
            model.load_state_dict(pretrain['state_dict'])

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters
        else:
            model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
            model.module.fc = model.module.fc.cuda()
            return model, model.parameters()

    elif opt.model == 'resnet2p1d':
        model = resnet2p1d.generate_model(model_depth=opt.model_depth, n_classes=opt.n_classes,
                                          n_input_channels=opt.input_channels, shortcut_type=opt.resnet_shortcut,
                                          conv1_t_size=opt.conv1_t_size, conv1_t_stride=opt.conv1_t_stride,
                                          no_max_pool=opt.no_max_pool, widen_factor=opt.resnet_widen_factor)

        model = nn.DataParallel(model, device_ids=None).cuda()

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path, map_location='cpu')
            assert opt.arch == pretrain['arch']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrain['state_dict'].items():
                name = 'module.' + k
                new_state_dict[name] = v
            for k, v in new_state_dict.items():
                print(k)
            model.load_state_dict(new_state_dict)
            model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
            model.module.fc = model.module.fc.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

        elif opt.resume_path1:
            pretrain = torch.load(opt.resume_path1)
            assert opt.arch == pretrain['arch']
            #
            # model.load_state_dict(pretrain['state_dict'])
            # parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrain.items():
                name = 'module.' + k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
            model.module.fc = model.module.fc.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters
    elif opt.model == 'resnet':
        model = resnet.generate_model(model_depth=opt.model_depth,
                                      n_classes=opt.n_classes,
                                      n_input_channels=3,
                                      shortcut_type=opt.resnet_shortcut,
                                      conv1_t_size=opt.conv1_t_size,
                                      conv1_t_stride=opt.conv1_t_stride,
                                      no_max_pool=opt.no_max_pool,
                                      widen_factor=opt.resnet_widen_factor)
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path, map_location='cpu')
            assert opt.arch == pretrain['arch']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrain['state_dict'].items():
                name = 'module.' + k
                new_state_dict[name] = v
            for k, v in new_state_dict.items():
                print(k)
            model.load_state_dict(new_state_dict)
            model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
            model.module.fc = model.module.fc.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

        elif opt.resume_path1:
            pretrain = torch.load(opt.resume_path1)
            assert opt.arch == pretrain['arch']

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrain.items():
                name = 'module.' + k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
            model.module.fc = model.module.fc.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters
    elif opt.model == 's3dg':
        model = s3dg.S3DG(gating=True, slow=True)
        model = model.cuda()
        model = nn.DataParallel(model)
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)

            # assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])
            model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
            model.module.fc = model.module.fc.cuda()

            parameters = s3dg.get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters
    else:
        raise ValueError("Model {} is not support yet.".format(opt.model))

    return model, model.parameters()


def generate_model_resnext_18(opt):

    from resnext import get_fine_tuning_parameters
    model = resnext.resnet18(
        num_classes=opt.n_classes,
        shortcut_type=opt.resnet_shortcut,
        cardinality=opt.resnext_cardinality,
        sample_size=opt.sample_size,
        sample_duration=opt.sample_duration,
        input_channels=opt.input_channels,
        output_layers=opt.output_layers)

    model = model.cuda()
    model = nn.DataParallel(model)

    if opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path, map_location='cpu')
        assert opt.arch == pretrain['arch']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in pretrain.items():
            name = 'module.' + k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
        model.module.fc = model.module.fc.cuda()

        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        return model, parameters

    elif opt.resume_path1:
        pretrain = torch.load(opt.resume_path1)
        assert opt.arch == pretrain['arch']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in pretrain.items():
            name = 'module.' + k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
        model.module.fc = model.module.fc.cuda()

        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        return model, parameters

    return model, model.parameters()
