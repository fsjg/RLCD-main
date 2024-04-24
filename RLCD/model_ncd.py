import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion_ncd import Fusion
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class Net(nn.Module):
    def __init__(self, args, local_map):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim + 1
        self.prednet_len1, self.prednet_len2 = 512, 256

        super(Net, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.knowledge_emb = nn.Embedding(self.knowledge_dim, self.knowledge_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.knowledge_dim) 
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.e_guess = nn.Embedding(self.emb_num,1)

        self.k_index = torch.LongTensor(list(range(self.stu_dim))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.emb_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_n))).to(self.device)

        self.FusionLayer1 = Fusion(args, local_map)
        self.FusionLayer2 = Fusion(args, local_map)

        self.prednet_full1 = nn.Linear(2 * args.knowledge_n, 1, bias=False)
        self.prednet_full2 = nn.Linear(2 * args.knowledge_n, 1, bias=False)
        
        self.fc1 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias = True)
        self.fc2 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias = True)
        self.sigmoid = nn.Sigmoid()
        
        self.prednet_1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_r,train_count):
        all_stu_emb = self.student_emb(self.stu_index).to(self.device)
        exer_emb = self.exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.knowledge_emb(self.k_index).to(self.device)

        # Fusion layer 1
        kn_emb1, exer_emb1, all_stu_emb1 = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb) 
        # Fusion layer 2
        kn_emb2, exer_emb2, all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)

        # get batch student data
        batch_stu_emb = all_stu_emb2[stu_id] # 128 123
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0], batch_stu_emb.shape[1], batch_stu_emb.shape[1])

        # get batch exercise data
        batch_exer_emb = exer_emb2[exer_id]  # 128 123
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0], batch_exer_emb.shape[1], batch_exer_emb.shape[1])

        # get batch knowledge concept data
        kn_vector = kn_emb2.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], kn_emb.shape[0], kn_emb.shape[1]) # [128,123,123]

        # Cognitive diagnosis
        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2))).squeeze(2) # [128,123]
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2))).squeeze(2)  # [128,123]
        
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10  # [128,1]
        e_guess=torch.sigmoid(self.e_guess(stu_id)) * 10 # [128,1]
        
        input_x = kn_r * (preference - diff) * e_discrimination
        input_x = torch.cat((input_x, e_guess),1)
        input_x = self.drop_1(torch.sigmoid(self.prednet_1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_2(input_x)))
        output = torch.sigmoid(self.prednet_3(input_x))

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_1.apply(clipper)
        self.prednet_2.apply(clipper)
        self.prednet_3.apply(clipper)
        self.fc1.apply(clipper)
        self.fc2.apply(clipper)
    
    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)