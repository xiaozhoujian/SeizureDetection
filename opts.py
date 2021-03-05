from argparse import ArgumentParser

__all__ = ["parse_args"]


def parse_args():
    # Path arguments
    parser = ArgumentParser(description="Seizure detection of mice")
    path_args = parser.add_argument_group("Path Arguments")
    path_args.add_argument("--source_dir", type=str, required=True,
                           help="Path of the source data")
    path_args.add_argument("--output_dir", type=str, required=True,
                           help="Path to store intermediate and result files")
    path_args.add_argument("--annotation_path", default="mice_labels/class.txt", type=str,
                           help="File contains label information")
    path_args.add_argument("--resume_path", default="results_mice_resnext101/save_82.pth", type=str,
                           help="The trained model file.")
    path_args.add_argument("--result_name", default="result", type=str,
                           help="The name of result folder under the output dir")

    # Preprocess arguments
    pre_args = parser.add_argument_group("Preprocess Arguments")
    pre_args.add_argument("--expert", type=str, required=True,
                          help="The name of the expert, used to rename videos.")
    pre_args.add_argument("--subject_name", type=str, required=True,
                          help="The name of subject, used to rename videos.")
    pre_args.add_argument("--mul_num", default=10, type=int,
                          help="Number of multiprocessing")

    # network arguments
    net_args = parser.add_argument_group("Network arguments")
    net_args.add_argument("--classes", default=2, type=int,
                          help="number of classes output in the network")
    net_args.add_argument("--dataset", default="mice_online", type=str,
                          help="name of the dataset")
    net_args.add_argument("--model", default="resnext", type=str,
                          help="name of the model")
    net_args.add_argument("--model_depth", default=101, type=int,
                          help="the depth of the network")
    net_args.add_argument("--sample_duration", default=64, type=int,
                          help="the clips number of the network input")
    net_args.add_argument("--resnet_shortcut", default="b", type=str)
    net_args.add_argument("--resnet_cardinality", default=32, type=int)
    net_args.add_argument("--sample_size", default=112, type=int,
                          help="the clip size of the network input [sample_size, sample_size]")
    net_args.add_argument("--use_cuda", default=True, type=bool)
    net_args.add_argument("--arch", default="resnext-101", type=str,
                          help="the specific name of the network architecture")
    net_args.add_argument("--batch_size", default=1, type=int,
                          help="the input batch size of the validation")
    net_args.add_argument("--num_workers", default=8, type=int,
                          help="the number of workers in the data loader")
    net_args.add_argument("--input_channels", default=3, type=int,
                          help="the number of channel of videos")

    # Pipeline arguments
    pipline_args = parser.add_argument_group("Pipeline arguments")
    pipline_args.add_argument("--preprocess", default=False, action="store_true",
                              help="Flag to check if we need to preprocess data.")
    pipline_args.add_argument("--predict", default=False, action="store_true",
                              help="Flag to check if we need to predict the preprocessed data")
    pipline_args.add_argument("--remove_intermediate", default=False, action="store_true",
                              help="Flag to check if we eneed to remove the intermediate files")
    pipline_args.add_argument("--threshold", default=0.3, type=float,
                              help="Threshold value for post processing, only effect when --post_process is given.")
    pipline_args.add_argument("--post_process", default=False, action="store_true",
                              help="Flag to check if we need to post process the result from our network")
    pipline_args.add_argument("--svm", default=False, action="store_true",
                              help="Flag to check if we need to use the svm to further classify the network result")

    args = parser.parse_args()
    return args
