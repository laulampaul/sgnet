This is the original implementation of the paper joint demosaicing and denoising with self guidance.

run . script/run_sgnet-dn-ps200.sh for training and . script/resr_sgnet-dn-ps200.sh for testing


result:
When the sigma=5,
             Dense texture   Sparse texture        MIT moire        Urban100
             PSNR LPIPS        PSNR LPIPS      PSNR SSIM LPIPS  PSNR SSIM LPIPS
SGNet   46.75 0.01433     47.88 0.01917  32.15 0.9043 0.0691  34.54 0.9533 0.0299
