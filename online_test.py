from opts import parse_opts
from torch import nn
from train_val_min import iterate_dataloader, initialization


if __name__ == "__main__":
    opts = parse_opts()
    print(opts)
    opts.input_channels = 3

    model, val_data, val_dataloader, val_logger = initialization(opts, is_train=False)
    softmax = nn.Softmax(dim=1)
    iterate_dataloader(val_dataloader, opts, model, softmax, epoch=None, val_file_name=opts.val_file_1,
                       val_logger=val_logger, is_train=False)
