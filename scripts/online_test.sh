python online_test.py --learning_rate 0.002 --weight_decay 1e-5 --dataset MICE --modality RGB \
--split 1 --only_RGB --n_classes 400 --n_finetune_classes 2 \
--batch_size 16 --log 1 --sample_duration 64 --model resnext \
--ft_begin_index 0 \
--annotation_path "dataset/mice_labels" --result_path "results/online_test" \
--resume_path1 "pretrained_model/save_17.pth" \
--n_workers 32 --n_epochs 100 \
--val_file_1 'online_test.txt' \
--val_path_1 '/media/ntk/WD_BLACK_3/K25_videos/pre_case'