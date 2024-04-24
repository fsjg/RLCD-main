# -*- coding: utf-8 -*-

import dgl
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def build_graph(type, node):
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(node)
    edge_list = []  
    if type == 'direct':
        with open('RLCD-main/data/junyi/graph/K_Directed.txt', 'r') as f:
            for line in f.readlines():
                # 这里与RCD处理的txt不同
                line = line.split()
                edge_list.append((int(line[0]), int(line[1])))

        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'undirect':
        with open('RLCD-main/data/junyi/graph/K_Undirected.txt', 'r') as f:
            for line in f.readlines():
                line = line.split()
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        # edges are directional in DGL; make them bi-directional
        g.add_edges(dst, src)
        return g
    elif type == 'similar_exer':
        with open('RLCD-main/data/junyi/graph/similar.txt', 'r') as f:
            for line in f.readlines():
                line = line.split()
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        g.add_edges(dst, src)   
        # 一跳
        adjacency_matrix_1 = dgl.khop_adj(g, 1).numpy()
        # 二跳
        adjacency_matrix_2 = dgl.khop_adj(g, 2).numpy()
        # 三跳
        adjacency_matrix_3 = dgl.khop_adj(g, 3).numpy()
        r, c = adjacency_matrix_2.shape
        adjacency_matrix_2 = np.where(adjacency_matrix_2 != 0, 1, 0)
        adjacency_matrix_3 = np.where(adjacency_matrix_3 != 0, 1, 0)
        # 或运算 True表示相关, False表示不相关
        ad_mask = np.logical_or(adjacency_matrix_1, adjacency_matrix_2, adjacency_matrix_3).astype(np.bool) 
        # 对角线要变成False
        ad = ~np.eye(r, dtype=np.bool)  # 对角矩阵取反
        relation_mask = np.logical_and(ad_mask, ad).astype(np.float)  # 相关节点mask 不包括自身
        unrelation_mask = np.logical_and(~ad_mask, ad).astype(np.float)  # 不相关节点mask 不包括自身
        return g, (relation_mask, unrelation_mask) # 计算习题间余弦相似度
        # return g
    elif type == 'similar_stu':
        with open('RLCD-main/data/junyi/graph/same_teacher.txt', 'r') as f:
            for line in f.readlines():
                line = line.split()
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        g.add_edges(dst, src)
        return g
