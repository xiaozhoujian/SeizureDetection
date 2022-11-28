python train_val_min.py --learning_rate 0.002 --weight_decay 1e-5 --dataset MICE --modality RGB \
--split 1 --only_RGB --n_classes 400 --n_finetune_classes 2 \
--batch_size 16 --log 1 --sample_duration 64 --model resnext \
--ft_begin_index 0 --frame_dir "/media/ntk/WD_BLACK_3/K25_frames" \
--annotation_path "dataset/mice_labels" --pretrain_path "pretrained_model/RGB_Kinetics_64f.pth" --result_path "results/K25" \
--n_workers 32 --n_epochs 3 --train_file 'K25_train.txt' \
--val_file_1 'K25_val_case.txt' --val_file_2 'K25_val_control.txt' \
--val_path_1 '/media/ntk/WD_BLACK_3/K25_videos' --val_path_2 '/media/ntk/WD_BLACK_3/K25_videos'
