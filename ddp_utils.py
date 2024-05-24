import os
from time import sleep
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
import random

def init_fn(worker_id):
    np.random.seed(987)
    random.seed(987)

def process_group_initialized():
    try:
        dist.get_world_size()
        return True
    except:
        return False

def calc_num_workers():
    try:
        world_size = dist.get_world_size()
    except:
        world_size = 1
    return len(os.sched_getaffinity(0)) // world_size

def setup_ddp(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.manual_seed(987)
    torch.cuda.set_device(rank)

def init_ddp():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(11451 + np.random.randint(100))
    world_size = torch.cuda.device_count()
    assert world_size > 0, "You need a GPU!"
    smp = mp.get_context('spawn')
    return smp, world_size

def wait_for_world(state: mp.Queue, world_size):
    state.put(1)
    while state.qsize() < world_size:
        pass
    for _ in range(world_size):
        state.get()