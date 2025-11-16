import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure Chinese characters render correctly
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC'] + plt.rcParams['font.sans-serif']
sns.set_style('whitegrid', {'font.sans-serif': ['Noto Sans CJK SC', 'sans-serif']})


def plot_top_words(output_dir: str, vocab: list, beta_matrix: np.ndarray,
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
        ax.set_title(f"主题 {k+1}：Top {topn} 词")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.suptitle(title, fontsize=14)
    out_path = os.path.join(output_dir, 'top_words.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_prevalence_sentiment(output_dir: str, A_opt: np.ndarray, covs: np.ndarray,
                               K: int, label: str = '演示'):    
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
        ax.set_xlabel('评分（stars）')
        ax.set_ylabel('主题情感潜变量')
        ax.set_title(f'主题 {k+1}：情感 vs 评分（{label}）')
    fig.suptitle('主题情感与评分关系示意', fontsize=14)
    out_path = os.path.join(output_dir, 'sentiment_vs_rating.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path
