import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv('K_Directed.csv')
df2 = pd.read_csv('K_Undirected.csv')

# 拼接两个数据框
result = pd.concat([df1, df2])
sorted_result = result.sort_values(result.columns[0])

# 将拼接后的结果保存为一个新的CSV文件
sorted_result.to_csv('output.csv', index=False)
