import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from fusion_ncd import Fusion
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class Net(nn.Module):
    def __init__(self, args, local_map):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim

        super(Net, self).__init__()

        # network struct ure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.knowledge_emb = nn.Embedding(self.knowledge_dim, self.knowledge_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.knowledge_dim)

        self.k_index = torch.LongTensor(list(range(self.stu_dim))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.emb_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_n))).to(self.device)

        self.FusionLayer1 = Fusion(args, local_map)
        self.FusionLayer2 = Fusion(args, local_map)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self):
        all_stu_emb = self.student_emb(self.stu_index).to(self.device)
        exer_emb = self.exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.knowledge_emb(self.k_index).to(self.device)

        # Fusion layer 1
        kn_emb1, exer_emb1, all_stu_emb1 = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)
        # Fusion layer 2
        kn_emb2, exer_emb2, all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)

        return exer_emb

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
