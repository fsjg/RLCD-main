import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphLayer import GraphLayer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class Fusion(nn.Module):
    def __init__(self, args, local_map):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim

        # graph structure
        self.directed_g = local_map['directed_g'].to(self.device)
        self.undirected_g = local_map['undirected_g'].to(self.device)
        self.similar_exer_g = local_map['similar_exer_g'].to(self.device)
        self.similar_stu_g = local_map['similar_stu_g'].to(self.device)

        super(Fusion, self).__init__()

        self.directed_gat = GraphLayer(self.directed_g, args.knowledge_n, args.knowledge_n)
        self.undirected_gat = GraphLayer(self.undirected_g, args.knowledge_n, args.knowledge_n)
        self.similar_exer_gat = GraphLayer(self.similar_exer_g, args.knowledge_n, args.knowledge_n)
        self.similar_stu_gat = GraphLayer(self.similar_stu_g, args.knowledge_n, args.knowledge_n)

        self.k_attn_fc1 = nn.Linear(2 * args.knowledge_n, 1, bias=True)
        self.k_attn_fc2 = nn.Linear(2 * args.knowledge_n, 1, bias=True)

    def forward(self, kn_emb, exer_emb, all_stu_emb):
        k_directed = self.directed_gat(kn_emb)
        k_undirected = self.undirected_gat(kn_emb)
        similar_exer = self.similar_exer_gat(exer_emb)
        similar_stu = self.similar_stu_gat(all_stu_emb)

        # update concepts
        A = kn_emb
        B = k_directed
        C = k_undirected
        concat_c_1 = torch.cat([A, B], dim=1)
        concat_c_2 = torch.cat([A, C], dim=1)
        score1 = self.k_attn_fc1(concat_c_1)
        score2 = self.k_attn_fc2(concat_c_2)
        score = F.softmax((torch.cat([score1, score2], dim=1)),dim=1)  # dim = 1, 按行SoftMax, 行和为1
        kn_emb = A + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C

        # updated exercises
        exer_emb = exer_emb + similar_exer

        # updated students
        all_stu_emb = all_stu_emb + similar_stu

        return kn_emb, exer_emb, all_stu_emb
