# SEA-RAFT
This repository contains the source code for our paper SEA-RAFT, which is a fast, simple, and effective framework for optical flow estimation. On [Spring](https://spring-benchmark.org/) benchmark, SEA-RAFT can process 1080p image pairs at >20FPS while achieving state-of-the-art performance.

<img src="assets/visualization.png" width='1000'>

If you find SEA-RAFT useful for your work, please consider citing our academic paper:

<h3 align="center">
    <a href="https://arxiv.org/abs/2405.14793">
        SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow
    </a>
</h3>
<p align="center">
    <a href="https://memoryslices.github.io/">Yihan Wang</a>, 
    <a href="https://www.lahavlipson.com/">Lahav Lipson</a>, 
    <a href="http://www.cs.princeton.edu/~jiadeng">Jia Deng</a><br>
</p>

```
@article{wang2024sea,
  title={SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow},
  author={Wang, Yihan and Lipson, Lahav and Deng, Jia},
  journal={arXiv preprint arXiv:2405.14793},
  year={2024}
}
```

## Requirements
Our code is developed with pytorch 2.2.0, CUDA 12.2 and python 3.10.
```Shell
conda create --name SEA-RAFT python=3.10.13
conda activate SEA-RAFT
pip install -r requirements.txt
```

## Model Zoo
Please download the models from [google drive](https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW?usp=sharing) and put them into the `models` folder.

## Required Data
To evaluate/train SEA-RAFT, you will need to download the required datasets. 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/)
* [TartanAir](https://theairlab.org/tartanair-dataset/)
* [Spring](https://spring-benchmark.org/)

By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder. Please check [RAFT](https://github.com/princeton-vl/RAFT) for more details.

```Shell
├── datasets
    ├── Sintel
    ├── KITTI
    ├── FlyingChairs/FlyingChairs_release
    ├── FlyingThings3D
    ├── HD1K
    ├── spring
        ├── test
        ├── train
        ├── val
    ├── tartanair
```

## Training, Evaluation, and Submission

Please refer to [scripts/train.sh](scripts/train.sh), [scripts/eval.sh](scripts/eval.sh), and [scripts/submission.sh](scripts/submission.sh) for more details.

## Acknowledgements

This project would not have been possible without relying on some awesome repos: [RAFT](https://github.com/princeton-vl/RAFT), [unimatch](https://github.com/autonomousvision/unimatch/tree/master), [Flowformer](https://github.com/drinkingcoder/FlowFormer-Official), [ptlflow](https://github.com/hmorimitsu/ptlflow), and [LoFTR](https://github.com/zju3dv/LoFTR). We thank the original authors for their excellent work.