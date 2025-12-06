import os
import random
import numpy as np
from typing import List, Dict

from sts_repro.sts.data import Document, Corpus, build_vocabulary, vectorize_corpus
from sts_repro.sts.anchor import AnchorInitializer
from sts_repro.sts.model import STSModel
from sts_repro.sts.em import EMRunner
from sts_repro.sts.viz import plot_top_words, plot_prevalence_sentiment

"""
Demo runner:
- Try to load Yelp Review Full (stars 1-5) via datasets; if unavailable, fallback to a small local sample.
- Build vocabulary and doc-term matrix
- Initialize STS with anchor words
- Run EM to estimate latent prevalence/sentiment & parameters
- Produce visualizations
"""

OUTPUT_DIR = 'output'


def load_reviews(n_docs: int = 1000) -> List[Document]:
    try:
        import pandas as pd
        # 读取test.csv文件
        df = pd.read_csv('data/Hotel_Reviews.csv')

        # 确保Negative_Review和Positive_Review为字符串类型，缺失值用空字符串填充
        df['Negative_Review'] = df['Negative_Review'].fillna('').astype(str)
        df['Positive_Review'] = df['Positive_Review'].fillna('').astype(str)

        # 合并正负评论作为文本内容
        df['text'] = df['Negative_Review'] + df['Positive_Review']

        # 截取指定数量的文档（不超过数据集总数）
        n_docs = min(n_docs, len(df))
        df = df.head(n_docs)

        docs: List[Document] = []
        for _, row in df.iterrows():
            # 使用Reviewer_Score作为stars（1-10范围）
            stars = float(row['Reviewer_Score'])
            covs = {
                'intercept': 1.0,
                'stars': stars
            }
            docs.append(Document(text=row['text'], covariates=covs))
        return docs
    except Exception as e:
        print(f"Error loading Yelp Review Full: {e}")
        # 保留原有的 fallback 机制
        samples = [
            ("Great food and friendly staff. Loved the pizza!", 5),
            ("Service was slow, but the burger tasted fine.", 3),
            ("Horrible experience. Dirty tables and cold fries.", 1),
            ("Amazing pasta, cozy atmosphere. Will come again.", 5),
            ("Average coffee. Price is okay.", 3),
            ("Terrible service. Manager was rude.", 1),
            ("Nice patio seating and fresh salad.", 4),
            ("Waited too long. Food arrived cold.", 2),
            ("Delicious tacos! Highly recommend.", 5),
            ("Mask policy not followed. Very disappointing.", 1),
            ("Good takeout experience, fast pickup.", 4),
            ("Mediocre taste, decent price.", 3),
        ]
        docs = []
        for t, s in samples:
            docs.append(Document(text=t, covariates={'intercept': 1.0, 'stars': float(s)}))
        return docs


def to_matrix(docs: List[Document]) -> np.ndarray:
    keys = sorted(list(docs[0].covariates.keys()))
    X = np.zeros((len(docs), len(keys)))
    for i, d in enumerate(docs):
        for j, k in enumerate(keys):
            X[i, j] = d.covariates[k]
    return X, keys


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    docs = load_reviews(n_docs=5000)  # 取前500条信息
    texts = [d.text for d in docs]
    # 设置每条评论最大长度以及最小词频
    # vocab = build_vocabulary(texts, max_vocab=2000, min_df=5)
    vocab = build_vocabulary(texts, max_vocab=2000, min_df=2)
    dtm = vectorize_corpus(docs, vocab)
    X, cov_keys = to_matrix(docs)
    # 设置主题数K
    K = 6
    model = STSModel(K=K, V=len(vocab), ix=X.shape[1], random_state=0)
    # Anchor initializer
    ai = AnchorInitializer(K=K, vocab_size=len(vocab))
    beta0, anchors = ai.initialize(dtm)
    runner = EMRunner(model, max_em_iter=10, tol=1e-4, group_size=50)
    runner.initialize_kappa_with_anchor(beta0)
    A_opt, Phis = runner.run(dtm, X)
    # Visualizations
    # Average beta per topic using A_opt median a_s
    a_s_med = np.median(A_opt[:, K:(2 * K)], axis=0)
    beta_mat = np.zeros((K, len(vocab)))
    for k in range(K):
        beta_mat[k] = model.beta_topic(k, a_s_med[k])
    top_words_path = plot_top_words(OUTPUT_DIR, vocab, beta_mat, title='STS', picname='test5k')
    senti_path = plot_prevalence_sentiment(
        output_dir=OUTPUT_DIR,
        A_opt=A_opt,
        covs=X,
        K=K,
        picname='test5k',
        label='Hotel Presentation'
    )
    print('Artifacts saved:', top_words_path, senti_path)


if __name__ == '__main__':
    main()
