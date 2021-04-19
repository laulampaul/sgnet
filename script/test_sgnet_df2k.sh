#!/usr/bin/env bash

python test.py --pretrained_model your/save/path --model sgnet --scale 1 --bias --denoise --sigma 0 \
--test_path ../TENet_sim_test/input/urban100 --save_path output --crop_scale 8 --postname df2k


# if Run out of CUDA memory, just set --crop_scale 4 (or higher)
# remember to change sigma according to the noise level of the current image
