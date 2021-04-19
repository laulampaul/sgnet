#!/usr/bin/env bash
python test.py --pretrained_model your/save/path --scale 1 --model sgnet --denoise --sigma 0 --postname ps200 --crop_scale 8 --show_info \
--test_path /cache/liulin/pixelshift/PixelShift200_val_crop --save_path output


