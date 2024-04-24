import torch
import torch.nn as nn
import sys
from model_1 import Net
import torch.nn.functional as F
from utils import CommonArgParser, construct_local_map
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def train(args, local_map):
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    net = Net(args, local_map)
    net = net.to(device)
    embedding = net.forward()
    
    diff = embedding.unsqueeze(1) - embedding.unsqueeze(0)
    # 计算欧氏距离
    euclidean_distance = torch.norm(diff, dim=2)
    
    dist = euclidean_distance.detach().cpu().numpy()
    dist1 = (dist[617][520]+dist[354][355]+dist[284][705])/3
    dist2 = (dist[586][528]+dist[805][707]+dist[805][138])/3
    print(dist[617][520])
    print(dist[586][528])
    print(dist1)
    print(dist2)
    print(dist[617][521])


if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    local_map = construct_local_map(args)
    train(args, local_map)