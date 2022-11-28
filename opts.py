import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    # Datasets 
    parser.add_argument(
        '--frame_dir',
        default='dataset/HMDB51/',
        type=str,
        help='path of jpg files')
    parser.add_argument(
        '--annotation_path',
        default='dataset/HMDB51_labels',
        type=str,
        help='label paths')
    parser.add_argument(
        '--dataset',
        default='HMDB51',
        type=str,
        help='(HMDB51, UCF101, Kinectics)')
    parser.add_argument(
        '--split',
        default=1,
        type=str,
        help='(for HMDB51 and UCF101)')
    parser.add_argument(
        '--modality',
        default='RGB',
        type=str,
        help='(RGB, Flow)')
    parser.add_argument(
        '--input_channels',
        default=3,
        type=int,
        help='(3, 2)')
    parser.add_argument(
        '--n_classes',
        default=2,
        type=int,
        help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument(
        '--n_finetune_classes',
        default=51,
        type=int,
        help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    parser.add_argument(
        '--only_RGB',
        action='store_true',
        help='Extracted only RGB frames')
    parser.set_defaults(only_RGB=False)

    # Model parameters
    parser.add_argument(
        '--output_layers',
        action='append',
        help='layer to output on forward pass')
    parser.set_defaults(output_layers=[])
    parser.add_argument(
        '--model',
        default='resnext',
        type=str,
        help='Model base architecture')
    parser.add_argument(
        '--model_depth',
        default=101,
        type=int,
        help='Number of layers in model')
    parser.add_argument('--conv1_t_size',
                        default=7,
                        type=int,
                        help='Kernel size in t dim of conv1.')
    parser.add_argument('--conv1_t_stride',
                        default=1,
                        type=int,
                        help='Stride in t dim of conv1.')
    parser.add_argument('--no_max_pool',
                        action='store_true',
                        help='If true, the max pooling after conv1 is removed.')
    parser.add_argument(
        '--resnet_widen_factor',
        default=1.0,
        type=float,
        help='The number of feature maps of resnet is multiplied by this value')
    parser.add_argument('--wide_resnet_k',
                        default=2,
                        type=int,
                        help='Wide resnet k')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--resnext_cardinality',
        default=32,
        type=int,
        help='ResNeXt cardinality')
    parser.add_argument(
        '--ft_begin_index',
        default=4,
        type=int,
        help='Begin block index of fine-tuning')
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=64,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--training',
        action='store_true',
        help='training/testing')
    parser.add_argument(
        '--amp',
        default='True',
        type=str)
    parser.set_defaults(training=True)
    parser.add_argument(
        '--freeze_BN',
        action='store_true',
        help='freeze_BN/testing')
    parser.set_defaults(freeze_BN=False)
    parser.add_argument(
        '--batch_size',
        default=256,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '--n_workers',
        default=32,
        type=int,
        help='Number of workers for dataloader')

    # optimizer parameters
    parser.add_argument(
        '--learning_rate',
        default=0.008,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        help='Momentum')
    parser.add_argument(
        '--dampening',
        default=0.9,
        type=float,
        help='dampening of SGD')
    parser.add_argument(
        '--weight_decay',
        default=1e-5,
        type=float,
        help='Weight Decay')
    parser.add_argument(
        '--nesterov',
        action='store_true',
        help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        help='Currently only support SGD')
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument(
        '--MARS_alpha',
        default=50,
        type=float,
        help='Weight of Flow augemented MSE loss')
    parser.add_argument(
        '--n_epochs',
        default=400,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')

    # options for logging
    parser.add_argument(
        '--result_path',
        default='',
        type=str,
        help='result_path')
    parser.add_argument(
        '--MARS',
        action='store_true',
        help='test MARS')
    parser.set_defaults(MARS=False)
    parser.add_argument(
        '--pretrain_path',
        default='',
        type=str,
        help='Pretrained model (.pth)')
    parser.add_argument(
        '--MARS_pretrain_path',
        default='',
        type=str,
        help='Pretrained model (.pth)')
    parser.add_argument(
        '--MARS_resume_path',
        default='',
        type=str,
        help='MARS resume model (.pth)')
    parser.add_argument(
        '--resume_path1',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--resume_path2',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--resume_path3',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--log',
        default=1,
        type=int,
        help='Log training and validation')
    parser.add_argument(
        '--checkpoint',
        default=2,
        type=int,
        help='Trained model is saved at every this epochs.')

    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument(
        '--random_seed', default=1, type=bool, help='Manually set random seed of sampling validation clip')
    parser.add_argument(
        '--highest_val', default={'name': 0}, type=dict, help='store best val')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='whether resume of not')
    parser.add_argument(
        '--inputs',
        default='',
        type=str,
        help='inputs video for real time show')
    parser.add_argument(
        '--train_file',
        default='',
        type=str,
        help='file name of train file')
    parser.add_argument(
        '--test_file',
        default='',
        type=str,
        help='file name of test file')
    parser.add_argument(
        '--val_file_1',
        default='',
        type=str,
        help='file name of val file')
    parser.add_argument(
        '--val_file_2',
        default='',
        type=str,
        help='file name of val file')
    parser.add_argument(
        '--val_path_1',
        default='',
        type=str,
        help='path of val file')
    parser.add_argument(
        '--val_path_2',
        default='',
        type=str,
        help='path of val file')
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        '--local_rank',
        default=-1,
        type=int,
        help='GPU ranks for data distributed parallel')
    parser.add_argument(
        '--rank',
        default=1,
        type=int,
        help='Processes ranks for data distributed parallel')
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--nprocs', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--distributed', action='store_true', help='Whether distributed, auto set in code')
    parser.add_argument('--sync_bn', default=1, type=int, help='0-NOT use syncBN, 1-use syncBN')
    parser.add_argument('--clip_grad_norm', default=1, type=int, help='0-NOT use clip_grad_norm, 1-use clip_grad_norm')
    parser.add_argument('--split_path', default="", type=str, help='path of traning dataset')
    parser.add_argument('--pb_rate', default=4, type=int, help='playback rate of a clip 1,2,4,8')
    parser.add_argument('--transform_mode', default="numpy", type=str, help="Transform mode numpy/ft_train")
    parser.add_argument('--input_size', default=320, type=int, help="Input size")
    parser.add_argument('--output_feat', default=128, type=int, help="Output feature size")
    parser.add_argument('--norm_method', default="tf_norm", type=str, help="norm method for inputs: tf_norm, imagenet")
    parser.add_argument('--max_iter', default=80000, type=int, help="Maximum iteration")
    parser.add_argument('--loss_weight', default=1.0, nargs="+", type=float, help="loss contrastive/speed1/speed2")
    parser.add_argument('--t_ft_task', default="", type=str, help="Finetune task for test")
    parser.add_argument('--sc_type', default="B", type=str, help="shortcut type for resnet")

    # jojen add argument
    parser.add_argument("--input_")
    args = parser.parse_args()

    return args
