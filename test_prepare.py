import pandas as pd

# 读取CSV文件（请替换为你的文件路径）
df = pd.read_csv('sts_repro/data/Hotel_Reviews.csv')

# 提取前200条数据
test_df = df.head(10000)

# 保存为新的CSV文件（test集）
test_df.to_csv('sts_repro/data/test.csv', index=False)

print("已成功保存前200条数据为test.csv")