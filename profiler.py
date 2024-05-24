import sys
sys.path.append('core')
import argparse
import torch
from config.parser import parse_args
from raft import RAFT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args = parse_args(parser)
    model = RAFT(args)
    model.eval()
    h, w = [540, 960]
    input = torch.zeros(1, 3, h, w)
    model = model.cuda()
    input = input.cuda()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        with_flops=True) as prof:
            output = model(input, input, iters=args.iters, test_mode=True)
    
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=5))
    events = prof.events()
    forward_MACs = sum([int(evt.flops) for evt in events])
    print("forward MACs: ", forward_MACs / 2 / 1e9, "G")

if __name__ == '__main__':
    main()