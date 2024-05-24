import sys
sys.path.append('core')

import argparse
import os
import cv2
import numpy as np

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

import datasets
from raft import RAFT
from tqdm import tqdm

from utils.flow_viz import flow_to_image
from utils import frame_utils
from utils.utils import load_ckpt, InputPadder

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down

@torch.no_grad()
def create_spring_submission(args, model, output_path='../spring_submission'):
    """ Create submission for the Sintel leaderboard """
    test_dataset = datasets.SpringFlowDataset(split='test', aug_params=None)
    args = args_list[0]
    pbar = tqdm(total=len(test_dataset))
    for test_id in range(len(test_dataset)):
        image1, image2, extra_info = test_dataset[test_id]
        frame, scene, cam, direction = extra_info
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        flow, info = calc_flow(args, model, image1, image2)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        flow_gt_vis = flow_to_image(flow, convert_to_bgr=True)
        output_dir = os.path.join(output_path, scene, f"flow_{direction}_{cam}")
        output_file = os.path.join(output_dir, f"flow_{direction}_{cam}_{frame:04d}.flo5")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv2.imwrite(os.path.join(output_dir, f"flow_{direction}_{cam}_{frame:04d}.png"), flow_gt_vis)
        frame_utils.writeFlo5File(flow, output_file)
        pbar.update(1)

    pbar.close()

@torch.no_grad()
def create_sintel_submission(args, model, output_path='../sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        flow_prev, sequence_prev = None, None
        pbar = tqdm(total=len(test_dataset))
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            flow, info = calc_flow(args, model, image1, image2)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            flow_gt_vis = flow_to_image(flow, convert_to_bgr=True)
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            cv2.imwrite(os.path.join(output_dir, f"frame{frame+1}.png"), flow_gt_vis)
            sequence_prev = sequence
            pbar.update(1)
        
        pbar.close()

@torch.no_grad()
def create_kitti_submission(args, model, output_path='../kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    test_dataset = datasets.KITTI(split='testing', aug_params=None)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    pbar = tqdm(total=len(test_dataset))
    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        flow, info = calc_flow(args, model, image1, image2)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        output_filename = os.path.join(output_path, frame_id)
        flow_gt_vis = flow_to_image(flow, convert_to_bgr=True)
        cv2.imwrite(os.path.join(output_path, f"frame{frame_id}"), flow_gt_vis)
        frame_utils.writeFlowKITTI(output_filename, flow)
        pbar.update(1)

    pbar.close()

def eval(args):
    args.gpus = [0]
    model = RAFT(args)
    load_ckpt(model, args.model)
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        if args.dataset == 'spring':
            create_spring_submission(args, model, output_path='../spring_submission')
        elif args.dataset == 'sintel':
            create_sintel_submission(args, model, output_path='../sintel_submission')
        elif args.dataset == 'kitti':
            create_kitti_submission(args, model, output_path='../kitti_submission')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='checkpoint path', required=True, type=str)
    args = parse_args(parser)
    eval(args)


if __name__ == '__main__':
    main()
