import torch
import torch.nn as nn
import sys
from model_1 import Net
import numpy as np
import pandas as pd
import torch.nn.functional as F
from utils import CommonArgParser, construct_local_map
import os
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(args, local_map):
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    net = Net(args, local_map)
    net = net.to(device)
    embeddings = net.forward()

    # Example tensor with shape (n_users, 2)
    # tensor = exer_emb2.cpu().detach().numpy()

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv("/mnt/home/E21301289/RCD-main/data/ASSIST17/graph/K_Undirected.csv")

    # Convert the DataFrame to PyTorch tensors
    users = torch.tensor(df["user"].values)
    similar_users = torch.tensor(df["similar_user"].values)

    # Load the user embeddings into a PyTorch tensor
    print(embeddings.shape)

    # Calculate the Euclidean distance between each pair of similar users
    distances = torch.norm(embeddings[users] - embeddings[similar_users], dim=1)

    # Calculate the average distance
    average_distance = distances.mean()

    print(f"Average distance between similar users: {average_distance}")


if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    local_map = construct_local_map(args)
    train(args, local_map)