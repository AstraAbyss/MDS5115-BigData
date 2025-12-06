#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate paired comparison charts between STS hotwords and BERTopic hotwords
per mapped topic. Save figures to experiments/embedding_topics/figs/paired_topic_{k}_vs_{kp}.png
Also optionally generate a quality metrics figure if available.
"""
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- Config paths ---
STS_CSV = "sts_hotel_sample_1k/outputs/topic_hotwords_sts.csv"
BERT_CSV = "experiments/embedding_topics/outputs/topic_hotwords_bertopic.csv"
MAP_CSV = "experiments/embedding_topics/outputs/topic_mapping.csv"
MET_CSV = "experiments/embedding_topics/outputs/topic_quality_metrics.csv"
FIG_DIR = "experiments/embedding_topics/figs"
OUT_OVERLAP = "experiments/embedding_topics/outputs/paired_topic_sentiment_overlap.csv"

os.makedirs(FIG_DIR, exist_ok=True)

# --- Fonts for Chinese ---
plt.rcParams['font.family'] = 'sans-serif'
# Prefer Noto Sans CJK SC, fallback to Source Han Sans SC
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Source Han Sans SC', 'Source Han Sans SC VF', 'DejaVu Sans'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- Read data ---
sts = pd.read_csv(STS_CSV)
ber = pd.read_csv(BERT_CSV)
mapdf = pd.read_csv(MAP_CSV)

# Normalize BERTopic scores to [0,1] within each topic_id & sentiment_setting
ber['norm_score'] = ber.groupby(['topic_id', 'sentiment_setting'])['score'].transform(lambda s: (s / (s.max() if s.max() else 1.0)))

# Build mapping: for each sts_topic_id pick bertopic_topic_id with highest cosine_similarity
mapdf = mapdf.copy()
if 'cosine_similarity' in mapdf.columns:
    metric_col = 'cosine_similarity'
elif 'match_score' in mapdf.columns:
    metric_col = 'match_score'
else:
    # fallback to jaccard_avg
    metric_col = 'jaccard_avg'

best_map_rows = mapdf.sort_values(metric_col, ascending=False).drop_duplicates(subset=['sts_topic_id'])
# mapping dict: sts_topic_id -> (bertopic_topic_id, match_score)
best_map = {}
for _, row in best_map_rows.iterrows():
    best_map[int(row['sts_topic_id'])] = (int(row['bertopic_topic_id']), float(row[metric_col]))

sentiments = ['average', 'negative', 'positive']

# Prepare overlap output
overlap_records = []

# Helper: draw horizontal bar chart
def draw_barh(ax, words, values, title, color):
    # reverse for horizontal bar (small at bottom)
    words_plot = list(words)[::-1]
    values_plot = list(values)[::-1]
    ax.barh(range(len(words_plot)), values_plot, color=color)
    ax.set_yticks(range(len(words_plot)))
    ax.set_yticklabels(words_plot, fontsize=9)
    ax.set_xlim(0, max(values_plot) * 1.10 if len(values_plot) else 1)
    ax.set_title(title, fontsize=11)
    # numeric formatting for percentages if value range within 0-1
    if max(values_plot) <= 1.0:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*100:.0f}%"))
    ax.tick_params(axis='x', labelsize=9)

# Iterate STS topics
for sts_topic in sorted(sts['topic_id'].unique()):
    if sts_topic not in best_map:
        # skip if no mapping
        continue
    bert_topic, match_score = best_map[sts_topic]

    # Create figure with 3 rows x 2 columns
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), dpi=200, constrained_layout=True)

    for i, sent in enumerate(sentiments):
        # STS side
        sts_slice = sts[(sts['topic_id'] == sts_topic) & (sts['sentiment_setting'] == sent)].sort_values('rank')
        # Ensure Top-15 only
        sts_slice = sts_slice.head(15)
        draw_barh(axes[i,0], sts_slice['word'], sts_slice['probability'], f"STS - {sent} Top-15", color="#4C78A8")

        # BERTopic side
        ber_slice = ber[(ber['topic_id'] == bert_topic) & (ber['sentiment_setting'] == sent)].sort_values('rank')
        ber_slice = ber_slice.head(15)
        draw_barh(axes[i,1], ber_slice['word'], ber_slice['norm_score'], f"BERTopic - {sent} Top-15（归一化）", color="#F58518")

        # Compute Jaccard overlap and common words
        stw = set(sts_slice['word'].str.lower().tolist())
        bew = set(ber_slice['word'].str.lower().tolist())
        inter = stw & bew
        union = stw | bew
        jacc = (len(inter) / len(union)) if len(union) > 0 else 0.0
        overlap_records.append({
            'sts_topic_id': sts_topic,
            'bertopic_topic_id': bert_topic,
            'sentiment_setting': sent,
            'jaccard_overlap': jacc,
            'common_words': ", ".join(sorted(inter))
        })

    fig.suptitle(f"Topic {sts_topic} ↔ Topic {bert_topic}（match_score={match_score:.3f}）", fontsize=14)
    out_path = os.path.join(FIG_DIR, f"paired_topic_{sts_topic}_vs_{bert_topic}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# Save overlap csv
pd.DataFrame(overlap_records).to_csv(OUT_OVERLAP, index=False)

# Quality metrics figure (if available)
if os.path.exists(MET_CSV):
    met = pd.read_csv(MET_CSV)
    # Only BERTopic metrics available, make two subplots
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), dpi=200, constrained_layout=True)
    # Coherence
    ax[0].bar(met['topic_id'].astype(str).tolist(), met['coherence_cv'], color="#4C78A8")
    ax[0].set_title("BERTopic 主题连贯性（Coherence CV）", fontsize=13)
    ax[0].set_ylabel("Coherence")
    ax[0].tick_params(axis='x', rotation=0)
    # Exclusivity
    ax[1].bar(met['topic_id'].astype(str).tolist(), met['exclusivity'], color="#F58518")
    ax[1].set_title("BERTopic 主题排他性（Exclusivity）", fontsize=13)
    ax[1].set_ylabel("Exclusivity")
    ax[1].tick_params(axis='x', rotation=0)
    fig.suptitle("主题质量指标（仅 BERTopic；STS 指标缺失）", fontsize=14)
    plt.savefig(os.path.join(FIG_DIR, "topic_quality_comparison.png"), bbox_inches="tight")
    plt.close(fig)

print(f"Done. Figures saved to: {FIG_DIR}")
