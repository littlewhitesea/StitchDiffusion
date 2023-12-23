# StitchDiffusion (Keep Update)
Customizing 360-Degree Panoramas through Text-to-Image Diffusion Models \
[Hai Wang](https://littlewhitesea.github.io/), [Xiaoyu Xiang](https://engineering.purdue.edu/people/xiaoyu.xiang.1), [Yuchen Fan](https://ychfan.github.io/), [Jing-Hao Xue](https://www.homepages.ucl.ac.uk/~ucakjxu/)

[![Project](https://img.shields.io/badge/Project-Website-orange)](https://littlewhitesea.github.io/stitchdiffusion.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2310.18840-b31b1b.svg)](https://arxiv.org/abs/2310.18840)

### [Data](https://drive.google.com/file/d/1EgRwj5BqO7Y-PvdL8mrFwKsqmgN_N4_b/view?usp=sharing) | [Pretrained Model](https://drive.google.com/file/d/1MiaG8v0ZmkTwwrzIEFtVoBj-Jjqi_5lz/view?usp=sharing)

[Colab](https://github.com/lshus/stitchdiffusion-colab) was implemented by @[lshus](https://github.com/lshus). 

## StitchDiffusion Code

Actually, StitchDiffusion is a tailored generation (denoising) process for synthesizing 360-degree panoramas, we provide its core code here.

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

```python
#####################
## StitchDiffusion ##
#####################

views_t = get_views(height, width) # height = 512; width = 4*height = 2048
count_t = torch.zeros_like(latents)
value_t = torch.zeros_like(latents)
# latents are sampled from standard normal distribution (torch.randn()) with a size of Bx4x64x256,
# where B denotes the batch size.

for i, t in enumerate(tqdm(timesteps)):

    count_t.zero_()
    value_t.zero_()

    # initialize the value of latent_view_t
    latent_view_t = latents[:, :, :, 64:192]

    #### pre-denoising operations twice on the stitch block ####
    for ii_md in range(2):

        latent_view_t[:, :, :, 0:64] = latents[:, :, :, 192:256] #left part of the stitch block
        latent_view_t[:, :, :, 64:128] = latents[:, :, :, 0:64] #right part of the stitch block

        # expand the latents if we are doing classifier free guidance
        latent_model_input = latent_view_t.repeat((2, 1, 1, 1))

        # # predict the noise residual
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the denoising step with the reference (customized) model
        latent_view_denoised = self.scheduler.step(noise_pred, t, latent_view_t)['prev_sample']

        value_t[:, :, :, 192:256] += latent_view_denoised[:, :, :, 0:64]
        count_t[:, :, :, 192:256] += 1

        value_t[:, :, :, 0:64] += latent_view_denoised[:, :, :, 64:128]
        count_t[:, :, :, 0:64] += 1

    # same denoising operations as what MultiDiffusion does
    for h_start, h_end, w_start, w_end in views_t:

        latent_view_t = latents[:, :, h_start:h_end, w_start:w_end]
    
        # expand the latents if we are doing classifier free guidance
        latent_model_input = latent_view_t.repeat((2, 1, 1, 1))
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

        #perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the denoising step with the reference (customized) model
        latent_view_denoised = self.scheduler.step(noise_pred, t, latent_view_t)['prev_sample']
        value_t[:, :, h_start:h_end, w_start:w_end] += latent_view_denoised
        count_t[:, :, h_start:h_end, w_start:w_end] += 1

    latents = torch.where(count_t > 0, value_t / count_t, value_t)

latents = 1 / 0.18215 * latents
image = self.vae.decode(latents).sample
image = (image / 2 + 0.5).clamp(0, 1)


#### global cropping operation ####
image = image[:, :, :, 512:1536]
image = image.cpu().permute(0, 2, 3, 1).float().numpy()
```

## Useful Tools

[360 panoramic images viewer](https://renderstuff.com/tools/360-panorama-web-viewer/): It could be used to view the synthesized 360-degree panorama.

[Seamless Texture Checker](https://www.pycheung.com/checker/): It could be employed to check the continuity between the leftmost and rightmost sides of the generated image. 

[clip-interrogator](https://github.com/pharmapsychotic/clip-interrogator?tab=readme-ov-file): It contains Google Colab of BLIP to generate text prompts.

## Statement
This research was done by Hai Wang in University College London. The code and released models are owned by Hai Wang.

## Citation
If you find the code helpful in your research or work, please cite our paper:
```Bibtex
@inproceedings{wang2024customizing,
  title={Customizing 360-Degree Panoramas through Text-to-Image Diffusion Models},
  author={Wang, Hai and Xiang, Xiaoyu and Fan, Yuchen and Xue, Jing-Hao},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={4933--4943},
  year={2024}
}
```
## Acknowledgments
We thank [MultiDiffusion](https://github.com/omerbt/MultiDiffusion). Our work is based on their excellent codes.
