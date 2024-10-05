# SEA-RAFT

[[Paper](https://arxiv.org/abs/2405.14793)][[Slides](https://docs.google.com/presentation/d/1xZn-NowHuPqfdLDAaQwKyzYvP4HzGmT7/edit?usp=sharing&ouid=118125745783453356964&rtpof=true&sd=true)]

We introduce SEA-RAFT, a more simple, efficient, and accurate [RAFT](https://github.com/princeton-vl/RAFT) for optical flow. Compared with RAFT, SEA-RAFT is trained with a new loss (mixture of Laplace). It directly regresses an initial flow for faster convergence in iterative refinements and introduces rigid-motion pre-training to improve generalization. SEA-RAFT achieves state-of-the-art accuracy on the [Spring benchmark](https://spring-benchmark.org/) with a 3.69 endpoint-error (EPE) and a 0.36 1-pixel outlier rate (1px), representing 22.9\% and 17.8\% error reduction from best-published results. In addition, SEA-RAFT obtains the best cross-dataset generalization on KITTI and Spring. With its high efficiency, SEA-RAFT operates at least 2.3x faster than existing methods while maintaining competitive performance.

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

Google Drive: [link](https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW?usp=sharing).

HuggingFace: [link](https://huggingface.co/papers/2405.14793).

## Custom Usage

We provide an example in `custom.py`. By default, this file will take two RGB images as the input and provide visualizations of the optical flow and the uncertainty. You can load your model by providing the path:
```Shell
python custom.py --cfg config/eval/spring-M.json --path models/Tartan-C-T-TSKH-spring540x960-M.pth
```
or load our models through HuggingFace🤗 (make sure you have installed huggingface-hub):
```Shell
python custom.py --cfg config/eval/spring-M.json --url MemorySlices/Tartan-C-T-TSKH-spring540x960-M
```

## Datasets
To evaluate/train SEA-RAFT, you will need to download the required datasets: [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs), [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [Sintel](http://sintel.is.tue.mpg.de/), [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow), [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/), [TartanAir](https://theairlab.org/tartanair-dataset/), and [Spring](https://spring-benchmark.org/).

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

This project relies on code from existing repositories: [RAFT](https://github.com/princeton-vl/RAFT), [unimatch](https://github.com/autonomousvision/unimatch/tree/master), [Flowformer](https://github.com/drinkingcoder/FlowFormer-Official), [ptlflow](https://github.com/hmorimitsu/ptlflow), and [LoFTR](https://github.com/zju3dv/LoFTR). We thank the original authors for their excellent work.
