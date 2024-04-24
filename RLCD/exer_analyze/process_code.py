import torch
import torch.nn as nn
import sys
from model_1 import Net
import torch.nn.functional as F
from utils import CommonArgParser, construct_local_map
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def EuclideanDistances(a,b):
    output = F.pairwise_distance(a, b, p=2)
    return output 


def analyze_cos(exer_feat, mask):
    """

    Args:
        exer_feat: type=Tensor, shape=[num_exer, dims]
        mask: type=tuple, len=2| relation_mask, unrelation_mask

    Returns:
        
    """
    relation_mask, unrelation_mask = torch.BoolTensor(mask[0]).cuda(), torch.BoolTensor(mask[1]).cuda()
    # 均一化, 目的是为了让每个习题向量的模转换成单位1
    exer_norm = exer_feat / torch.norm(exer_feat, dim=1, keepdim=True)
    # 计算每个习题之间的余弦相似度
    sim = torch.matmul(exer_norm, exer_norm.T)
    sim_rad = torch.acos(sim)
    
    # 利用mask提取出相关节点之间的余弦相似度 
    relation_sim = sim[relation_mask]
    # 利用取反后的mask提取出相关节点之间的余弦相似度
    unrelation_sim = sim[unrelation_mask]
    # 所有相关节点的相似度求均值（里面包含自己和自己的余弦相似度需要处理掉）
    rela = torch.max(relation_sim)
    # 所有不相关节点之间的相似度求均值
    unrela = torch.max(unrelation_sim)
    
    return rela.item(), unrela.item()

    

def train(args, local_map, mask):
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    net = Net(args, local_map)
    net = net.to(device)
    exer_emb2 = net.forward()
    
    seed = np.random.randint(low=0, high=exer_emb2.shape[0])
    # MIN, MAX = analyze(exer_emb2, mask)
    rela, unrela = analyze_cos(exer_emb2, mask)
    print("相关点之间余弦值 mean: {} \n 不相关点之间余弦值 mean: {}".format(rela, unrela))

if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    local_map, mask = construct_local_map(args)
    train(args, local_map, mask)