import torch
import torch.nn as nn
import sys
from model_ncd import Net
import torch.nn.functional as F
from utils import CommonArgParser, construct_local_map
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def train(args, local_map, mask):
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    net = Net(args, local_map)
    net = net.to(device)
    exer_emb2 = net.forward()

    # calculate the differences between all pairs of users
    diffs = exer_emb2[:, np.newaxis, :] - exer_emb2[np.newaxis, :, :]
    # diffs is now a tensor of shape (3, 3, 4)

    # square the differences element-wise
    diffs_squared = np.square(diffs)

    # sum the squared differences along the m dimension
    sum_squared_diffs = np.sum(diffs_squared, axis=2)

    # take the square root of the sum of squared differences element-wise
    euclidean_distances = np.sqrt(sum_squared_diffs)

    # euclidean_distances is now a tensor of shape (3, 3), where each element
    # represents the Euclidean distance between the corresponding users

    

if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    local_map, mask = construct_local_map(args)
    train(args, local_map, mask)