import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter # 导入格式化器

# --- 1. 数据准备 ---
# 请将 'your_data.csv' 替换为你的实际文件路径
df = pd.read_csv('data/sts_hotel_sample_1k_outputs_topic_hotwords_sts.csv')

# 确保数据按 'topic_id' 和 'sentiment' 排序，并在每个组内按 'probability' 降序排列
df.sort_values(by=['topic_id', 'sentiment_setting', 'probability'], ascending=[True, True, False], inplace=True)

# 获取唯一的 topics 和 sentiments，用于构建网格
topics = sorted(df['topic_id'].unique())
sentiments = sorted(df['sentiment_setting'].unique())

# 定义列数和行数：topics 的数量 + 1 列（用于显示 sentiment）
n_cols = len(topics) + 1
n_rows = len(sentiments)

# 定义每个单元格内显示的最大词语数量
max_words_per_cell = 8

# --- 2. 创建图表和布局 ---
# 增加 figure 的高度，为顶部标题留出空间
fig = plt.figure(figsize=(5 * n_cols, 0.8 * max_words_per_cell * n_rows + 1.5))

# 定义网格：n_rows + 1 行，n_cols 列。多出的一行用于放置 Topic 标题
gs = GridSpec(n_rows + 1, n_cols, figure=fig, 
              hspace=0.05,  # 增加垂直间距
              height_ratios=[0.2] + [1] * n_rows) # 第一行高度为0.2，其余行为1

# --- 3. 绘制 Topic 标题行 (新的位置) ---
for j, topic in enumerate(topics):
    ax = fig.add_subplot(gs[0, j]) # 在第一行绘制
    ax.text(0.5, 0.5, f'Topic {topic}', ha='center', va='center', 
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.axis('off')

# --- 4. 绘制主内容区 (热词和概率条) ---
# 预先计算全局最大概率，用于统一所有子图的x轴范围
global_max_prob = df['probability'].max()

for i, sentiment in enumerate(sentiments):
    for j, topic in enumerate(topics):
        cell_data = df[(df['topic_id'] == topic) & (df['sentiment_setting'] == sentiment)].head(max_words_per_cell)
        if cell_data.empty:
            continue
            
        ax = fig.add_subplot(gs[i + 1, j])
        
        bars = ax.barh(range(len(cell_data)), cell_data['probability'], left=0, color='lightblue', edgecolor='navy')
        
        for bar, word in zip(bars, cell_data['word']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                    word, ha='center', va='center', fontsize=9)

        ax.yaxis.set_visible(False)
        
        if i == n_rows - 1:
            ax.xaxis.set_visible(True)
            ax.set_xlim(0, global_max_prob * 1.1)
            ax.set_xticks(np.linspace(0, global_max_prob, 5))
            
            # --- 关键改动：设置X轴刻度为两位小数 ---
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            # 可选：旋转刻度标签以防止重叠
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        else:
            ax.xaxis.set_visible(False)
            
        for spine in ax.spines.values():
            spine.set_visible(False)

# --- 5. 绘制 Sentiment 列（旋转90度竖向+首字母大写） ---
for i, sentiment in enumerate(sentiments):
    ax = fig.add_subplot(gs[i + 1, -1])
    # 首字母大写处理
    sentiment_capped = sentiment.capitalize()
    # 核心改动：rotation=90 让文本旋转90度（竖向排列），调整对齐方式
    ax.text(0.5, 0.5, sentiment_capped, ha='center', va='center', 
            fontsize=12, fontweight='bold', transform=ax.transAxes, rotation=90)
    ax.axis('off')

# --- 6. 调整布局并显示 ---
fig.text(0.5, 0.01, 'Probability', ha='center', va='bottom', fontsize=16)
plt.suptitle('Topic-Sentiment Heatmap of Words', y=0.98, fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()
plt.savefig('data/sts_hotel_sample_1k_outputs_topic_hotwords_sts.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close()  # 关闭图表释放内存