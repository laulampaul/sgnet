#!/usr/bin/env bash
python -u train.py\
	--train_list datasets/train_df2k.txt --valid_list datasets/valid_df2k.txt\
    --model sgnet --bias --scale 1  --get2label\
    --denoise --max_noise 0.078 --min_noise 0.00
# max_noise 15: 0.06
# max_noise 10: 0.04
# max_noise 5: 0.02
# if run out of memory, lower batch_size down

