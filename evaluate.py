import sys
sys.path.append('core')
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

import datasets
from raft import RAFT
from tqdm import tqdm
from utils.utils import resize_data, load_ckpt

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
def validate_sintel(args, model):
    """ Peform validation using the Sintel (train) split """
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        val_loader = data.DataLoader(val_dataset, batch_size=8, 
            pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
        epe_list = np.array([], dtype=np.float32)
        px1_list = np.array([], dtype=np.float32)
        px3_list = np.array([], dtype=np.float32)
        px5_list = np.array([], dtype=np.float32)
        print(f"load data success {len(val_loader)}")
        for i_batch, data_blob in enumerate(val_loader):
            image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
            flow, info = calc_flow(args, model, image1, image2)
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
        print(f"Validation {dstype} EPE: {epe}, 1px: {100 * (1 - px1)}")


@torch.no_grad()
def validate_kitti(args, model):
    """ Peform validation using the KITTI-2015 (train) split """
    val_dataset = datasets.KITTI(split='training')
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    print(f"load data success {len(val_loader)}")
    epe_list = np.array([], dtype=np.float32)
    num_valid_pixels = 0
    out_valid_pixels = 0
    for i_batch, data_blob in enumerate(val_loader):
        image1, image2, flow_gt, valid_gt = [x.cuda(non_blocking=True) for x in data_blob]
        flow, info = calc_flow(args, model, image1, image2)
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
    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

@torch.no_grad()
def validate_spring(args, model):
    """ Peform validation using the Spring (val) split """
    val_dataset = datasets.SpringFlowDataset(split='val')
    val_loader = data.DataLoader(val_dataset, batch_size=4, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    
    epe_list = np.array([], dtype=np.float32)
    px1_list = np.array([], dtype=np.float32)
    px3_list = np.array([], dtype=np.float32)
    px5_list = np.array([], dtype=np.float32)
    print(f"load data success {len(val_loader)}")
    pbar = tqdm(total=len(val_loader))
    for i_batch, data_blob in enumerate(val_loader):
        image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
        flow, info = calc_flow(args, model, image1, image2)
        epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
        px1 = (epe < 1.0).float().mean(dim=[1, 2]).cpu().numpy()
        px3 = (epe < 3.0).float().mean(dim=[1, 2]).cpu().numpy()
        px5 = (epe < 5.0).float().mean(dim=[1, 2]).cpu().numpy()
        epe = epe.mean(dim=[1, 2]).cpu().numpy()
        epe_list = np.append(epe_list, epe)
        px1_list = np.append(px1_list, px1)
        px3_list = np.append(px3_list, px3)
        px5_list = np.append(px5_list, px5)
        pbar.update(1)

    pbar.close()
    epe = np.mean(epe_list)
    px1 = np.mean(px1_list)
    px3 = np.mean(px3_list)
    px5 = np.mean(px5_list)

    print(f"Validation Spring EPE: {epe}, 1px: {100 * (1 - px1)}")

@torch.no_grad()
def validate_middlebury(args, model):
    """ Peform validation using the Middlebury (public) split """
    val_dataset = datasets.Middlebury()
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    
    print(f"load data success {len(val_loader)}")
    epe_list = np.array([], dtype=np.float32)
    num_valid_pixels = 0
    out_valid_pixels = 0
    for i_batch, data_blob in enumerate(val_loader):
        image1, image2, flow_gt, valid_gt = [x.cuda(non_blocking=True) for x in data_blob]
        flow, info = calc_flow(args, model, image1, image2)
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
    args.gpus = [0]
    model = RAFT(args)
    load_ckpt(model, args.model)
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        if args.dataset == 'spring':
            validate_spring(args, model)
        elif args.dataset == 'sintel':
            validate_sintel(args, model)
        elif args.dataset == 'kitti':
            validate_kitti(args, model)
        elif args.dataset == 'middlebury':
            validate_middlebury(args, model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='checkpoint path', required=True, type=str)
    args = parse_args(parser)
    eval(args)

if __name__ == '__main__':
    main()

