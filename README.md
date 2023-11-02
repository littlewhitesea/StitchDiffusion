# StitchDiffusion (Keep Update)
Customizing 360-Degree Panoramas through Text-to-Image Diffusion Models \
[Hai Wang](https://littlewhitesea.github.io/), [Xiaoyu Xiang](https://engineering.purdue.edu/people/xiaoyu.xiang.1), [Yuchen Fan](https://ychfan.github.io/), [Jing-Hao Xue](https://www.homepages.ucl.ac.uk/~ucakjxu/)

[![arXiv](https://img.shields.io/badge/arXiv-2310.18840-b31b1b.svg)](https://arxiv.org/abs/2310.18840)
[![Project](https://img.shields.io/badge/Project-Website-orange)](https://littlewhitesea.github.io/stitchdiffusion.github.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/littlewhitesea/StitchDiffusion/blob/main/StitchDiffusion_360_Panorama.ipynb)

### [Paper](https://arxiv.org/pdf/2310.18840.pdf) | [Data](https://drive.google.com/file/d/1Iq1cRqhggrf8zWf4fHwf2hxkpNVw4kdF/view?usp=drive_link) | [Colab](https://colab.research.google.com/github/littlewhitesea/StitchDiffusion/blob/main/StitchDiffusion_360_Panorama.ipynb) 

## Introduction

Personalized text-to-image (T2I) synthesis based on diffusion models has attracted significant attention in recent research. However, existing methods primarily concentrate on customizing subjects or styles, neglecting the exploration of global geometry. In this study, we propose an approach that focuses on the customization of 360-degree panoramas, which inherently possess global geometric properties, using a T2I diffusion model. To achieve this, we curate a paired image-text dataset specifically designed for the task and subsequently employ it to fine-tune a pre-trained T2I diffusion model with LoRA. Nevertheless, the fine-tuned model alone does not ensure the continuity between the leftmost and rightmost sides of the synthesized images, a crucial characteristic of 360-degree panoramas. To address this issue, we propose a method called StitchDiffusion. Specifically, we perform pre-denoising operations twice at each time step of the denoising process on the stitch block consisting of the leftmost and rightmost image regions. Furthermore, a global cropping is adopted to synthesize seamless 360-degree panoramas. Experimental results demonstrate the effectiveness of our customized model combined with the proposed StitchDiffusion in generating high-quality 360-degree panoramic images. Moreover, our customized model exhibits exceptional generalization ability in producing scenes unseen in the fine-tuning dataset.

## Useful Tools

[360 panoramic images viewer](https://renderstuff.com/tools/360-panorama-web-viewer/): It could be used to view the synthesized 360-degree panorama.

[Seamless Texture Checker](https://www.pycheung.com/checker/): It could be employed to check the continuity between the leftmost and rightmost sides of the generated image. 

## Statement

This research was done by Hai Wang in University College London. The code and released models are owned by Hai Wang.

## Citation
If you find the code helpful in your research or work, please cite our paper:
```Bibtex
@misc{wang2023customizing,
      title={Customizing 360-Degree Panoramas through Text-to-Image Diffusion Models}, 
      author={Hai Wang and Xiaoyu Xiang and Yuchen Fan and Jing-Hao Xue},
      year={2023},
      eprint={2310.18840},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgments
We thank [MultiDiffusion](https://github.com/omerbt/MultiDiffusion) and [Kohya Trainer](https://github.com/Linaqruf/kohya-trainer). Our work is based on their excellent codes.
