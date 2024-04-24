import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model_ncd import Net
from utils import CommonArgParser, construct_local_map
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def train(args, local_map):
    data_loader = TrainDataLoader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(args, local_map)
    net = net.to(device)
    print (net)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    print('training model...')

    loss_function = nn.NLLLoss()
    for epoch in range(args.epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, train_count_emb = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, train_count_emb = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device), train_count_emb.to(device)
            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs, train_count_emb)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output+1e-10), labels)
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0

        # test and save current model every epoch
        save_snapshot(net, 'RLCD-main/model/model_epoch' + str(epoch + 1))
        rmse, auc = predict(args, net, epoch)

def predict(args, net, epoch):
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    data_loader = ValTestDataLoader('predict')
    proficiency = []
    print('predicting model...')
    data_loader.reset()
    net.eval()
    with torch.no_grad():
        correct_count, exer_count = 0, 0
        batch_count, batch_avg_loss = 0, 0.0
        pred_all, label_all = [], []
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, train_count_emb = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, train_count_emb = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device), labels.to(device), train_count_emb.to(device)
            output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs, train_count_emb)
            output = output.view(-1)
            # compute accuracy
            for i in range(len(labels)):
                proficiency.append(output[i])
                if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                    correct_count += 1
            
            exer_count += len(labels)
            pred_all += output.to(torch.device('cpu')).tolist()
            label_all += labels.to(torch.device('cpu')).tolist()

        pred_all = np.array(pred_all)
        label_all = np.array(label_all)
        # compute accuracy
        accuracy = correct_count / exer_count
        # compute RMSE
        rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
        # compute AUC
        auc = roc_auc_score(label_all, pred_all)
        print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
        with open('RLCD-main/result/ncd_model_val_junyi.txt', 'a', encoding='utf8') as f:
            f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))

        return rmse, auc

def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    local_map = construct_local_map(args)
    # local_map, mask = construct_local_map(args) # 计算习题间余弦相似度
    train(args, local_map)

    
