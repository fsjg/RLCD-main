import csv

# 打开纯文本文件
with open('K_Directed.txt', 'r') as txt_file:
    # 创建CSV文件并写入数据
    with open('K_Directed.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # 写入CSV头行
        csv_writer.writerow(['user', 'similar_user'])
        
        # 逐行读取纯文本文件并写入CSV文件
        for line in txt_file:
            # 分割行数据
            data = line.strip().split('\t')
            
            # 写入CSV行
            csv_writer.writerow(data)
