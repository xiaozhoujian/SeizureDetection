import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_name",
        default="v_ApplyEyeMakeup_g01_c01",
        type=str,
        help="the name of the video"
    )
    parser.add_argument(
        "--class_name",
        default="brush_hair",
        type=str,
        help="the name of the class (HMDB)"
    )
    parser.add_argument(
        "--dataset",
        default="UCF",
        type=str,
        help="the name of dataset"
    )
    parser.add_argument(
        "--siamese",
        action="store_true",
        help="whether to check siamese network"
    )
    parser.set_defaults(siamese=False)
    parser.add_argument(
        "--clip_class",
        default=0,
        type=int,
        help="the class of the clip"
    )
    parser.add_argument(
        "--cam_start",
        default=0,
        type=int,
        help="the class of the clip"
    )
    parser.add_argument(
        "--clip_s",
        default=20,
        type=int,
        help="start frame of the video"
    )
    parser.add_argument(
        "--clip_length",
        default=16,
        type=int,
        help="the length of the clip"
    )
    parser.add_argument(
        "--input_size",
        default=112,
        type=int,
        help="the size of the input"
    )
    parser.add_argument(
        "--gray",
        action="store_true",
        help="whether to use a gray image"
    )
    parser.set_defaults(gray=False)
    parser.add_argument(
        "--context",
        action="store_true",
        help="whether to use a gray image"
    )
    parser.set_defaults(context=False)
    parser.add_argument(
        "--video_num",
        default=[1, 1],
        nargs="+",
        type=int,
        help="video number of each class"
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="whether to use GPU")
    parser.add_argument(
        "--input_path",
        default="",
        type=str,
        help="path of input images"
    )
    parser.add_argument(
        "--buffer_capacity", default=10, type=int, help="the buffer capacity of RL"
    )
    parser.add_argument(
        "--steps_rl", default=2, type=int, help="the buffer capacity of RL"
    )
    parser.add_argument(
        "--batch_size", 
        default=32, 
        type=int, 
        help="Batch Size"
    )
    parser.add_argument(
        "--actions_num", default=3, type=int, help="the number of actions"
    )
    parser.add_argument(
        "--pre_actions_num", default=1, type=int, help="the number of pre actions in states"
    )
    parser.add_argument(
        "--show_img",
        action="store_true",
        help="whether to show cropped image")
    parser.add_argument(
        "--reward_rl",
        default="loss",
        type=str,
        help="reward of rl"
    )
    parser.add_argument(
        "--sample_size",
        default=112,
        type=int,
        help="Height and width of inputs")
    parser.add_argument(
        "--height", default=256, type=int, help="hight of image for loca tion net"
    )
    parser.add_argument(
        "--width", default=341, type=int, help="hight of image for location net"
    )
    parser.add_argument(
        "--sigma_mag", default=10, type=float, help="magnitude of sigma of RL agent"
    )
    parser.add_argument(
        "--device", default="", help="GPU device"
    )
    parser.add_argument(
        "--pretrain_path", default="", help="The path of pretrained model"
    )
    parser.add_argument("--cmaps", default="rainbow", help="The path of pretrained model")
    parser.add_argument("--frame_dir", default="", help="Frame dir")
    parser.add_argument("--input_file", default="", help="Frame vid list")
    parser.add_argument("--output_dir", default="", help="Frame vid list")
    args = parser.parse_args()
    
    return args

