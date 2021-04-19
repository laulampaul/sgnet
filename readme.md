
This is the implementation of the paper joint demosaicing and denoising with self guidance.

# Joint demosaicing and denoising with self guidance (CVPR'20)

Lin Liu, Xu Jia, Jianzhuang Liu, Qi Tian

[\[Paper Link\]](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Joint_Demosaicing_and_Denoising_With_Self_Guidance_CVPR_2020_paper.html) 

## Abstract
Usually located at the very early stages of the computational photography pipeline, demosaicing and denoising play important parts in the modern camera image processing. Recently, some neural networks have shown the effectiveness in joint demosaicing and denoising (JDD). Most of them first decompose a Bayer raw image into a four-channel RGGB image and then feed it into a neural network. This practice ignores the fact that the green channels are sampled at a double rate compared to the red and the blue channels. In this paper, we propose a self-guidance network (SGNet), where the green channels are initially estimated and then works as a guidance to recover all missing values in the input image. In addition, as regions of different frequencies suffer different levels of degradation in image restoration. We propose a density-map guidance to help the model deal with a wide range of frequencies. Our model outperforms state-of-the-art joint demosaicing and denoising methods on four public datasets, including two real and two synthetic data sets. Finally, we also verify that our method obtains best results in joint demosaicing , denoising and super-resolution.

## Reference
Our codes are based on TENet ( https://github.com/guochengqian/TENet ) and PSCNet (https://github.com/NVlabs/pacnet).

## Citation
If you use this code, please cite:

```
@InProceedings{Liu2020joint,
author = {Liu, Lin and Jia, Xu and Liu, Jianzhuang and Tian, Qi},
title = {Joint Demosaicing and Denoising With Self Guidance},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
