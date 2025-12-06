import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure Chinese characters render correctly
# plt.rcParams['font.family'] = 'Microsoft YaHei'
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC'] + plt.rcParams['font.sans-serif']
# sns.set_style('whitegrid', {'font.sans-serif': ['Noto Sans CJK SC', 'sans-serif']})


# 修正
# 1. 全局字体配置：优先使用微软雅黑（系统自带，无需额外安装）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimSun', 'sans-serif']
# 解释：
# - Microsoft YaHei：微软雅黑（优先）
# - SimSun：宋体（备选，防止微软雅黑缺失）
# - sans-serif：默认无衬线字体（最终 fallback）

# 2. 解决负号显示异常（避免图表中负号变成方块）
plt.rcParams['axes.unicode_minus'] = False

# 3. Seaborn 继承 Matplotlib 字体，无需重复设置 font.sans-serif
sns.set_style('whitegrid')  # 只保留样式，去掉字体配置


def plot_top_words(output_dir: str, vocab: list, beta_matrix: np.ndarray, picname: str,
                   title: str, topn: int = 12):
    """
    beta_matrix: K x V
    """
    os.makedirs(output_dir, exist_ok=True)
    K, V = beta_matrix.shape
    fig, axes = plt.subplots(K, 1, figsize=(12, 3*K), dpi=180, constrained_layout=True)
    for k in range(K):
        b = beta_matrix[k]
        idx = np.argsort(b)[-topn:][::-1]
        words = [vocab[i] for i in idx]
        vals = b[idx]
        ax = axes[k]
        ax.bar(words, vals, color='#5178c6')
        ax.set_title(f"Theme {k+1}:Top {topn} Words")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.suptitle(title, fontsize=14)
    out_path = os.path.join(output_dir, f'{picname}_top_words.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_prevalence_sentiment(output_dir: str, A_opt: np.ndarray, covs: np.ndarray, picname: str,
                               K: int, label: str = 'Pre'):
    os.makedirs(output_dir, exist_ok=True)
    # Show scatter of sentiment vs rating for first 4 topics if available
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180, constrained_layout=True)
    axes = axes.ravel()
    rating = covs[:, 1] if covs.shape[1] >= 2 else np.zeros(len(covs))
    for k in range(min(4, K)):
        ax = axes[k]
        sentiment_k = A_opt[:, K + k]
        # jitter rating to reduce overplotting
        jitter = rating + np.random.normal(0, 0.05, size=len(rating))
        ax.scatter(jitter, sentiment_k, s=6, alpha=0.5, color='#509863')
        ax.set_xlabel('Stars')
        ax.set_ylabel('Thematic Emotional Latent Variables')
        ax.set_title(f'Theme {k+1}:Emo vs Stars({label})')
    fig.suptitle('Schematic Relationship Between Theme Emotion and Rating', fontsize=14)
    out_path = os.path.join(output_dir, f'{picname}_sentiment_vs_rating.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path
