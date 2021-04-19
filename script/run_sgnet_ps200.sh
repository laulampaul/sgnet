#!/usr/bin/env bash
python -u train_mat.py \
	--train_list datasets/train_mat.txt --valid_list datasets/valid_mat.txt \
	--patch_size 128 --model sgnet --scale 2 --bias \
    --denoise --max_noise 0.06 --min_noise 0.00 --get2label
