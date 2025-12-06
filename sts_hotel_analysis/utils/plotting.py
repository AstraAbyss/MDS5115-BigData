import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

# Font configuration for Chinese
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC'] + plt.rcParams['font.sans-serif']


def plot_elbo_curve(elbo_list: List[float], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.plot(np.arange(len(elbo_list)), elbo_list, color='#2563eb', lw=2)
    ax.set_title('ELBO 收敛曲线')
    ax.set_xlabel('迭代次数', labelpad=10)
    ax.set_ylabel('ELBO 值', labelpad=10)
    plt.legend(['ELBO'], loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.25, top=0.85)
    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_score_relationships(theta: np.ndarray, alpha_s: np.ndarray, score_z: np.ndarray,
                             out_path: str) -> None:
    K = theta.shape[1]
    fig, axes = plt.subplots(K, 1, figsize=(10, 4*K), dpi=200)
    if K == 1:
        axes = [axes]
    for k in range(K):
        ax = axes[k]
        ax.scatter(score_z, theta[:, k], s=20, alpha=0.7, color='#10b981', label='主题占比')
        ax.scatter(score_z, alpha_s[:, k], s=20, alpha=0.5, color='#ef4444', label='话语/情感')
        ax.set_title(f'评分 vs 主题{ k+1 } 占比与话语情感')
        ax.set_xlabel('评分（z-score）', labelpad=10)
        ax.set_ylabel('数值', labelpad=10)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.25, top=0.9)
    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_triptype_diffs(theta: np.ndarray, alpha_s: np.ndarray, df: pd.DataFrame, trip_cols: List[str], out_path: str) -> None:
    # Compute averages per trip type
    K = theta.shape[1]
    rows = []
    for t in trip_cols:
        mask = df[f'trip_{t.replace(" ", "_")}'] == 1
        if mask.sum() == 0:
            continue
        avg_theta = theta[mask.values, :].mean(axis=0)
        avg_alpha = alpha_s[mask.values, :].mean(axis=0)
        for k in range(K):
            rows.append({'Trip_Type': t, 'Topic': f'T{k+1}', '占比': avg_theta[k], '话语情感': avg_alpha[k]})
    tab = pd.DataFrame(rows)
    # Heatmap-like bar chart per metric
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)
    sns.barplot(data=tab, x='Trip_Type', y='占比', hue='Topic', ax=axes[0])
    axes[0].set_title('不同旅行类型的主题占比')
    axes[0].tick_params(axis='x', rotation=45)
    sns.barplot(data=tab, x='Trip_Type', y='话语情感', hue='Topic', ax=axes[1])
    axes[1].set_title('不同旅行类型的话语情感')
    axes[1].tick_params(axis='x', rotation=45)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.3, top=0.85)
    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_time_trend(theta: np.ndarray, alpha_s: np.ndarray, days_z: np.ndarray, out_path: str) -> None:
    K = theta.shape[1]
    fig, axes = plt.subplots(K, 1, figsize=(10, 4*K), dpi=200)
    if K == 1:
        axes = [axes]
    for k in range(K):
        ax = axes[k]
        ax.scatter(days_z, theta[:, k], s=20, alpha=0.7, color='#3b82f6', label='主题占比')
        ax.scatter(days_z, alpha_s[:, k], s=20, alpha=0.5, color='#f59e0b', label='话语/情感')
        ax.set_title(f'时间变量 vs 主题{ k+1 } 占比与话语情感')
        ax.set_xlabel('days_since_review（z-score）', labelpad=10)
        ax.set_ylabel('数值', labelpad=10)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.25, top=0.9)
    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
