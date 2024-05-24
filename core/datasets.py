# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
import h5py
from tqdm import tqdm
from glob import glob
import os.path as osp
from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from utils.utils import induced_flow, check_cycle_consistency
from ddp_utils import *

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        self.dataset = 'unknown'
        self.subsample_groundtruth = False
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.mask_list = []
        self.extra_info = []

    def __getitem__(self, index):
        while True:
            try:
                return self.fetch(index)
            except Exception as e:
                index = random.randint(0, len(self) - 1)

    def fetch(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            if self.dataset == 'TartanAir':
                flow = np.load(self.flow_list[index])
                valid = np.load(self.mask_list[index])
                # rescale the valid mask to [0, 1]
                valid = 1 - valid / 100
                
            elif self.dataset == 'MegaDepth':
                depth0 = np.array(h5py.File(self.extra_info[index][0], 'r')['depth'])
                depth1 = np.array(h5py.File(self.extra_info[index][1], 'r')['depth'])
                camera_data = self.megascene[index]
                flow_01, flow_10 = induced_flow(depth0, depth1, camera_data)
                valid = check_cycle_consistency(flow_01, flow_10)
                flow = flow_01
            else:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            if self.dataset == 'Infinigen':
                # Inifinigen flow is stored as a 3D numpy array, [Flow, Depth]
                flow = np.load(self.flow_list[index])
                flow = flow[..., :2]
            else:
                flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        
        if self.subsample_groundtruth:
            # use only every second value in both spatial directions ==> flow will have same dimensions as images
            # used for spring dataset
            flow = flow[::2, ::2]

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        flow[torch.isnan(flow)] = 0
        flow[flow.abs() > 1e9] = 0

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1K'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1

class SpringFlowDataset(FlowDataset):
    """
    Dataset class for Spring optical flow dataset.
    For train, this dataset returns image1, image2, flow and a data tuple (framenum, scene name, left/right cam, FW/BW direction).
    For test, this dataset returns image1, image2 and a data tuple (framenum, scene name, left/right cam, FW/BW direction).

    root: root directory of the spring dataset (should contain test/train directories)
    split: train/test split
    subsample_groundtruth: If true, return ground truth such that it has the same dimensions as the images (1920x1080px); if false return full 4K resolution
    """
    def __init__(self, aug_params=None, root='datasets/spring', split='train', subsample_groundtruth=True):
        super(SpringFlowDataset, self).__init__(aug_params)

        assert split in ["train", "val", "test"]
        seq_root = os.path.join(root, split)

        if not os.path.exists(seq_root):
            raise ValueError(f"Spring {split} directory does not exist: {seq_root}")

        self.subsample_groundtruth = subsample_groundtruth
        self.split = split
        self.seq_root = seq_root
        self.data_list = []
        if split == 'test':
            self.is_test = True

        for scene in sorted(os.listdir(seq_root)):
            for cam in ["left", "right"]:
                images = sorted(glob(os.path.join(seq_root, scene, f"frame_{cam}", '*.png')))
                # forward
                for frame in range(1, len(images)):
                    self.data_list.append((frame, scene, cam, "FW"))
                # backward
                for frame in reversed(range(2, len(images)+1)):
                    self.data_list.append((frame, scene, cam, "BW"))

        for frame_data in self.data_list:
            frame, scene, cam, direction = frame_data

            img1_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame:04d}.png")

            if direction == "FW":
                img2_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame+1:04d}.png")
            else:
                img2_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame-1:04d}.png")

            self.image_list += [[img1_path, img2_path]]
            self.extra_info += [frame_data]

            if split != 'test':
                flow_path = os.path.join(self.seq_root, scene, f"flow_{direction}_{cam}", f"flow_{direction}_{cam}_{frame:04d}.flo5")
                self.flow_list += [flow_path]

class Infinigen(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/infinigen'):
        super(Infinigen, self).__init__(aug_params)
        self.root = root
        scenes = glob(osp.join(self.root, '*/*/'))
        self.dataset = "Infinigen"
        for scene in sorted(scenes):
            if not osp.isdir(osp.join(scene, 'frames')):
                continue
            images = sorted(glob(osp.join(scene, 'frames/Image/camera_0/*.png')))
            for idx in range(len(images) - 1):
                # name = Image + "_{ID}"
                ID = images[idx].split('/')[-1][6:-4]
                self.image_list.append([images[idx], images[idx + 1]])
                self.flow_list.append(osp.join(scene, 'frames/Flow3D/camera_0', f"Flow3D_{ID}.npy"))

class TartanAir(FlowDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, aug_params=None, root='datasets/tartanair'):
        super(TartanAir, self).__init__(aug_params, sparse=True)
        self.n_frames = 2
        self.dataset = 'TartanAir'
        self.root = root
        self._build_dataset()

    def _build_dataset(self):
        scenes = glob(osp.join(self.root, '*/*/*/*/*/*'))
        for scene in sorted(scenes):
            images = sorted(glob(osp.join(scene, 'image_left/*.png')))
            for idx in range(len(images) - 1):
                frame0 = str(idx).zfill(6)
                frame1 = str(idx + 1).zfill(6)
                self.image_list.append([images[idx], images[idx + 1]])
                self.flow_list.append(osp.join(scene, 'flow', f"{frame0}_{frame1}_flow.npy"))
                self.mask_list.append(osp.join(scene, 'flow', f"{frame0}_{frame1}_mask.npy"))

class MegaScene(data.Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 min_overlap_score=0.4,
                 **kwargs):

        super().__init__()
        self.root_dir = root_dir
        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]
        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
        depth_name0 = osp.join(self.root_dir, self.scene_info['depth_paths'][idx0])
        depth_name1 = osp.join(self.root_dir, self.scene_info['depth_paths'][idx1])
        # read intrinsics of original size
        K_0 = self.scene_info['intrinsics'][idx0].copy().reshape(3, 3)
        K_1 = self.scene_info['intrinsics'][idx1].copy().reshape(3, 3)
        # read and compute relative poses
        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        data = {
            'image0': img_name0,
            'image1': img_name1,
            'depth0': depth_name0,
            'depth1': depth_name1,
            'T0': T0,
            'T1': T1, # (4, 4)
            'K0': K_0,  # (3, 3)
            'K1': K_1,
        }
        return data

class MegaDepth(FlowDataset):

    def __init__(self, aug_params=None, root='datasets/megadepth'):
        super(MegaDepth, self).__init__(aug_params, sparse=True)
        self.n_frames = 2
        self.dataset = 'MegaDepth'
        self.root = root
        self._build_dataset()
        
    def _build_dataset(self):
        dataset_path = osp.join(self.root, 'train')
        index_folder = osp.join(self.root, 'index/scene_info_0.1_0.7')
        index_path_list = glob(index_folder + '/*.npz')
        dataset_list = []
        for index_path in index_path_list:
            my_dataset = MegaScene(dataset_path, index_path, min_overlap_score=0.4)
            dataset_list.append(my_dataset)
            
        self.megascene = torch.utils.data.ConcatDataset(dataset_list)
        for i in range(len(self.megascene)):
            data = self.megascene[i]
            self.image_list.append([data['image0'], data['image1']])
            self.extra_info.append([data['depth0'], data['depth1'], data['T0'], data['T1'], data['K0'], data['K1']])

class Middlebury(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/middlebury'):
        super(Middlebury, self).__init__(aug_params)
        img_root = os.path.join(root, 'images')
        flow_root = os.path.join(root, 'flow')

        flows = []
        imgs = []
        info = []

        for scene in sorted(os.listdir(flow_root)):
            img0 = os.path.join(img_root, scene, "frame10.png")
            img1 = os.path.join(img_root, scene, "frame11.png")
            flow = os.path.join(flow_root, scene, "flow10.flo") 
            imgs += [(img0, img1)]
            flows += [flow]
            info += [scene]

        self.image_list = imgs
        self.flow_list = flows
        self.extra_info = info

def fetch_dataloader(args, rank=0, world_size=1, use_ddp=False):
    """ Create the data loader for the corresponding trainign set """

    if args.dataset == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.1, 'max_scale': args.scale + 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.dataset == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.4, 'max_scale': args.scale + 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.dataset == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.6, 'do_flip': True}
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        train_dataset = sintel_clean + sintel_final

    elif args.dataset == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')
    
    elif args.dataset == 'spring':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale, 'max_scale': args.scale + 0.2, 'do_flip': True}
        train_dataset = SpringFlowDataset(aug_params, subsample_groundtruth=True)

    elif args.dataset == 'TartanAir':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.4, 'do_flip': True}
        train_dataset = TartanAir(aug_params)
    
    elif args.dataset == 'TSKH':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        kitti = KITTI({'crop_size': args.image_size, 'min_scale': args.scale - 0.3, 'max_scale': args.scale + 0.5, 'do_flip': True})
        hd1k = HD1K({'crop_size': args.image_size, 'min_scale': args.scale - 0.5, 'max_scale': args.scale + 0.2, 'do_flip': True})
        train_dataset = 20 * sintel_clean + 20 * sintel_final + 80 * kitti + 30 * hd1k + things
    
    elif args.dataset == 'TKH':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.4, 'max_scale': args.scale + 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        kitti = KITTI({'crop_size': args.image_size, 'min_scale': args.scale - 0.3, 'max_scale': args.scale + 0.5, 'do_flip': True})
        hd1k = HD1K({'crop_size': args.image_size, 'min_scale': args.scale - 0.5, 'max_scale': args.scale + 0.2, 'do_flip': True})
        train_dataset = 100 * hd1k + clean_dataset + final_dataset + 1000 * kitti

    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank)
        num_gpu = torch.cuda.device_count()
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size // num_gpu, 
            shuffle=(train_sampler is None), num_workers=calc_num_workers(), sampler=train_sampler, worker_init_fn=init_fn)
    else:
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
            pin_memory=False, shuffle=True, num_workers=32, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

