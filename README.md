# VertexShuffle-Based Spherical Super-Resolution for 360-Degree Videos

## Prerequisites

- Pytorch  
- Scikit-Learn 
- Numpy 
- FFmpeg
- Ligigl 

To generate mesh files used in the project, the Libigl library is required
https://libigl.github.io/

Use this command to install libigl
```sh
python -m pip install libigl
```

## Utils

``python gen_mesh.py``: generate full mesh, path: "./mesh_files/"

``python gen_partial.py``: generate our Focused Icosahedral Mesh, path: "./mesh_files/"

``python gen_rotation.py``: generate rotation matrics ``RM.npy`` for Focused Icosahedral Mesh


## Datasets
360-degree video head movement dataset

Dataset at: http://dash.ipv6.enstb.fr/headMovements/

Direct download link: http://dash.ipv6.enstb.fr/headMovements/archives/dataset.tar.gz


## Data Preparation

0. (optional) Generate video segments
```sh
ffmpeg -loglevel error -i INPUT -codec copy -flags +global_header -f segment -segment_time 1 -reset_timestamps 1 OUTPUT%05d.mp4
```

1. Generate image data, put image data into path "./VIDEONAME_img"

```sh
ffmpeg -i INPUT.mp4 %03d.png
```

2. Generate npy file, default_src path is "./VIDEONAME_img", default_dst path is "./VIDEONAME_data"
``to_npy.py``: line 49, 50, modify the src_path and dst_path
``python to_npy.py`` generate data for training and testing


## Usage


```sh
python train.py --model_idx 0 --feat 4 --batch-size 32 --test-batch-size 32 --epoch 15 --seed 0 --max_level 9 --min_level 7 --load 0 --video_name rollercoaster --lr 1e-2 --up_method SSR2
```

**model_idx**: segment model index for each video

**feat**: feature dimension

**up_method**: ['Trans','SSR1','SSR2'] stands for SSR with MeshConv transpose, VertexShuffle and VertexShuffle_V2, respectively

**load**: 1 stands for loading a pre-trained model, 0 stands for training from scratch

**video_name**: ['diving','paris','rollercoaster','timelapse','venise']

**max_level**: default 9, roughly equivalent to 2880x1440 resolution in 2D

**min_level**: [6,7], level-6 is roughly equivalent 360x180, level-7 is roughly equivalent 720x360. If set to 6, then upscale factor is x8, if set to 7, then upscale factor is x4. 6->9 is supported in SSR1 and SSR2 support only, no MeshConv transpose support.


## Citations

NOSSDAV'2022
```sh
@inproceedings{li2022applying,
  title={Applying VertexShuffle toward 360-degree video super-resolution},
  author={Li, Na and Liu, Yao},
  booktitle={Proceedings of the 32nd Workshop on Network and Operating Systems Support for Digital Audio and Video},
  pages={71--77},
  year={2022}
}
```

Transactions on Multimedia Computing Communications and Applications
```sh
To Appear
```

## License

MIT
