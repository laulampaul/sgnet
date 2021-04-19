#!/usr/bin/env bash
python -u test.py --pretrained_model checkpoints/checkpoints-tenet2-dn-matx6-6-6-64-2-rrdb-ps200 \
--scale 2 --bias --model tenet2 --denoise --sigma 0 --postname ps200 --crop_scale 8 --show_info \
--test_path /cache/liulin/pixelshift/PixelShift200_val_crop --save_path output


