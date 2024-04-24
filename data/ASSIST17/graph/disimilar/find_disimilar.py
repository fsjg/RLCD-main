import pandas as pd
import networkx as nx

# 超过两跳
# read the CSV file into a pandas dataframe
df = pd.read_csv('output.csv')

# create a directed graph using the user and similar_user columns
G = nx.from_pandas_edgelist(df, source='user', target='similar_user', create_using=nx.DiGraph())

# find all pairs of nodes that are more than two hops away from each other
pairs = []
for node in G.nodes:
    for neighbor in G.neighbors(node):
        for neighbor2 in G.neighbors(neighbor):
            if node != neighbor2 and not G.has_edge(node, neighbor2):
                pairs.append((node, neighbor2))

# write the results to a new CSV file
result_df = pd.DataFrame(pairs, columns=['user', 'similar_user'])
result_df.to_csv('dissimilar.csv', index=False)

# # 超过三跳
# # read the CSV file into a pandas dataframe
# df = pd.read_csv('output.csv')

# # create a directed graph using the user and similar_user columns
# G = nx.from_pandas_edgelist(df, source='user', target='similar_user', create_using=nx.DiGraph())

# # find all pairs of nodes that are more than three hops away from each other
# pairs = []
# for node in G.nodes:
#     for neighbor in G.neighbors(node):
#         for neighbor2 in G.neighbors(neighbor):
#             for neighbor3 in G.neighbors(neighbor2):
#                 if node != neighbor3 and not G.has_edge(node, neighbor3):
#                     pairs.append((node, neighbor3))

# write the results to a new CSV file
result_df = pd.DataFrame(pairs, columns=['user', 'similar_user'])
result_df.to_csv('dissimilar.csv', index=False)

