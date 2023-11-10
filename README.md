Code will come soon.

# StitchDiffusion (Keep Update)
Customizing 360-Degree Panoramas through Text-to-Image Diffusion Models \
[Hai Wang](https://littlewhitesea.github.io/), [Xiaoyu Xiang](https://engineering.purdue.edu/people/xiaoyu.xiang.1), [Yuchen Fan](https://ychfan.github.io/), [Jing-Hao Xue](https://www.homepages.ucl.ac.uk/~ucakjxu/)

[![Project](https://img.shields.io/badge/Project-Website-orange)](https://littlewhitesea.github.io/stitchdiffusion.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2310.18840-b31b1b.svg)](https://arxiv.org/abs/2310.18840)

### [Data](https://drive.google.com/file/d/1EgRwj5BqO7Y-PvdL8mrFwKsqmgN_N4_b/view?usp=sharing) | [Pretrained Model](https://drive.google.com/file/d/1MiaG8v0ZmkTwwrzIEFtVoBj-Jjqi_5lz/view?usp=sharing)

## StitchDiffusion Code

Since StitchDiffusion is a tailored generation process for synthesizing 360-degree panoramas, we provide its core code in the following.

```python
## following MultiDiffusion: https://github.com/omerbt/MultiDiffusion/blob/master/panorama.py ##
## the window size is changed for 360-degree panorama generation ##
def get_views(panorama_height, panorama_width, window_size=[64,128], stride=16):
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size[0]) // stride + 1
    num_blocks_width = (panorama_width - window_size[1]) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size[0]
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size[1]
        views.append((h_start, h_end, w_start, w_end))
    return views
```

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
We thank [MultiDiffusion](https://github.com/omerbt/MultiDiffusion). Our work is based on their excellent codes.
