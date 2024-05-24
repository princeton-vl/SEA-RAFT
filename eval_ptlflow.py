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
from utils.utils import resize_data, load_ckpt

import ptlflow
from ptlflow.utils import flow_utils

def forward_flow(model, image1, image2, scale=0, mode='downsample'):
    if mode == 'downsample':
        dlt = 0 # avoid edge effects
        image1 = image1 / 255.
        image2 = image2 / 255.
        img1 = F.interpolate(image1, scale_factor=2 ** scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** scale, mode='bilinear', align_corners=False)
        img1 = F.pad(img1, (dlt, dlt, dlt, dlt), "constant", 0)
        img2 = F.pad(img2, (dlt, dlt, dlt, dlt), "constant", 0)
        H, W = img1.shape[2:]
        inputs = {"images": torch.stack([img1, img2], dim=1)}
        predictions = model(inputs)
        flow = predictions['flows'][:, 0]
        flow = flow[..., dlt: H-dlt, dlt: W-dlt]
        flow = F.interpolate(flow, scale_factor=0.5 ** scale, mode='bilinear', align_corners=False) * (0.5 ** scale)
    else:
        raise NotImplementedError
    return flow

@torch.no_grad()
def validate_spring(model, mode='downsample'):
    """ Peform validation using the Spring (val) split """
    val_dataset = datasets.SpringFlowDataset(split='val') + datasets.SpringFlowDataset(split='train')
    val_loader = data.DataLoader(val_dataset, batch_size=4, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    
    epe_list = np.array([], dtype=np.float32)
    px1_list = np.array([], dtype=np.float32)
    px3_list = np.array([], dtype=np.float32)
    px5_list = np.array([], dtype=np.float32)
    for i_batch, data_blob in enumerate(val_loader):
        image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
        flow = forward_flow(model, image1, image2, scale=-1, mode=mode)
        epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
        px1 = (epe < 1.0).float().mean(dim=[1, 2]).cpu().numpy()
        px3 = (epe < 3.0).float().mean(dim=[1, 2]).cpu().numpy()
        px5 = (epe < 5.0).float().mean(dim=[1, 2]).cpu().numpy()
        epe = epe.mean(dim=[1, 2]).cpu().numpy()
        epe_list = np.append(epe_list, epe)
        px1_list = np.append(px1_list, px1)
        px3_list = np.append(px3_list, px3)
        px5_list = np.append(px5_list, px5)
        
    epe = np.mean(epe_list)
    px1 = np.mean(px1_list)
    px3 = np.mean(px3_list)
    px5 = np.mean(px5_list)

    print(f"Validation Spring EPE: {epe}, 1px: {100 * (1 - px1)}")

@torch.no_grad()
def validate_middlebury(model, mode='downsample'):
    """ Peform validation using the Middlebury (public) split """
    val_dataset = datasets.Middlebury()
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    epe_list = np.array([], dtype=np.float32)
    num_valid_pixels = 0
    out_valid_pixels = 0
    for i_batch, data_blob in enumerate(val_loader):
        image1, image2, flow_gt, valid_gt = [x.cuda(non_blocking=True) for x in data_blob]
        flow = forward_flow(model, image1, image2, scale=0, mode=mode)
        epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        val = valid_gt >= 0.5
        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        for b in range(out.shape[0]):
            epe_list = np.append(epe_list, epe[b][val[b]].mean().cpu().numpy())
            out_valid_pixels += out[b][val[b]].sum().cpu().numpy()
            num_valid_pixels += val[b].sum().cpu().numpy()
    
    epe = np.mean(epe_list)
    f1 = 100 * out_valid_pixels / num_valid_pixels
    print("Validation middlebury: %f, %f" % (epe, f1))

def eval(args):

    # Get an initialized model from PTLFlow
    device = torch.device('cuda')
    model = ptlflow.get_model(args.model, 'mixed').to(device)
    if "use_tile_input" in model.args:
        model.args.use_tile_input = False
    model.eval()
    print(args.model)
    with torch.no_grad():
        try:
            validate_middlebury(model, mode='downsample')
        except:
            print('Middlebury validation failed')
        validate_spring(model, mode='downsample')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='experiment configure file name', required=True, type=str)
    args = parser.parse_args()
    eval(args)

if __name__ == '__main__':
    main()