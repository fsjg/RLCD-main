import json
import torch


class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self):
        self.batch_size = 256
        self.ptr = 0
        self.data = []

        data_file = 'RLCD-main/data/junyi/train_set.json'
        config_file = 'RLCD-main/RLCD/config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            student_n, exercise_n, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)
        self.student_dim = int(student_n)
        self.exercise_dim = int(exercise_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys, train_count_emb = [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            train_count = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code-1] = 1.0
            if log['score'] > 0:
                for knowledge in log['knowledge_code']:
                    train_count[knowledge-1] += 1.0
            y = log['score']
            input_stu_ids.append(log['user_id']-1)
            input_exer_ids.append(log['exer_id']-1)
            input_knowledge_embs.append(knowledge_emb)
            train_count_emb.append(train_count)
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys), torch.Tensor(train_count_emb)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    def __init__(self, d_type='predict'):
        self.ptr = 0
        self.data = []
        self.d_type = d_type

        if d_type == 'predict':
            data_file = 'RLCD-main/data/junyi/test_set.json'
        config_file = 'RLCD-main/NCD/config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
            self.knowledge_dim = int(knowledge_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        logs = self.data[self.ptr]['logs']
        user_id = self.data[self.ptr]['user_id']
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys, train_count_emb = [], [], [], [], []
        for log in logs:
            input_stu_ids.append(user_id-1)
            input_exer_ids.append(log['exer_id']-1)
            knowledge_emb = [0.] * self.knowledge_dim
            train_count = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code-1] = 1.0
            if log['score'] > 0:
                for knowledge in log['knowledge_code']:
                    train_count[knowledge-1] += 1.0
            train_count_emb.append(train_count)
            input_knowledge_embs.append(knowledge_emb)
            y = log['score']
            ys.append(y)
        self.ptr += 1
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys), torch.Tensor(train_count_emb)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
