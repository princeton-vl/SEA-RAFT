import sys
sys.path.append('core')
import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import datasets
from raft import RAFT
from tqdm import tqdm

from utils import flow_viz
from utils import frame_utils
from utils.profile import profile_model
from utils.utils import resize_data, load_ckpt

import ptlflow
from ptlflow.utils import flow_utils

@torch.no_grad()
def eval(args):
    # Get an initialized model from PTLFlow
    model = ptlflow.get_model(args.model, 'mixed').cuda()
    if "use_tile_input" in model.args:
        model.args.use_tile_input = False
    model.eval()
    h, w = 540, 960
    inputs = {"images": torch.zeros(1, 2, 3, h, w).cuda()}
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU
        ],
        with_flops=True) as prof:
            output = model(inputs)
    events = prof.events()
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=5))
    forward_MACs = sum([int(evt.flops) for evt in events])
    print("forward MACs: ", forward_MACs / 2 / 1e9, "G")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='experiment configure file name', required=True, type=str)
    args = parser.parse_args()
    eval(args)

if __name__ == '__main__':
    main()