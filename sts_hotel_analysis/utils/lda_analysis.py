import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import statsmodels.api as sm
from nltk.sentiment.vader import SentimentIntensityAnalyzer


from .text_preprocess import build_corpus_and_covariates
from .plotting import plot_elbo_curve, plot_score_relationships, plot_triptype_diffs, plot_time_trend

def run_lda(dtm: np.ndarray, K: int, max_iter: int = 20, random_state: int = 0) -> LatentDirichletAllocation:
    lda = LatentDirichletAllocation(n_components=K, max_iter=max_iter, random_state=random_state,
                                   learning_method='online', learning_offset=50., n_jobs=-1)
    lda.fit(dtm)
    return lda

def select_K_lda(token_lists: list, grid: list) -> dict:
    texts = [" ".join(tokens) for tokens in token_lists]
    vectorizer = CountVectorizer(min_df=2)
    dtm = vectorizer.fit_transform(texts)
    
    results = []
    for k in grid:
        lda = run_lda(dtm, k)
        # Perplexity is a common metric for LDA
        perplexity = lda.perplexity(dtm)
        results.append({'K': k, 'perplexity': perplexity})
    
    res_df = pd.DataFrame(results)
    best_k = res_df.sort_values('perplexity', ascending=True).iloc[0]
    return {'best_K': int(best_k['K']), 'results': results, 'dtm': dtm, 'vectorizer': vectorizer}

def get_sentiment(df: pd.DataFrame):
    sid = SentimentIntensityAnalyzer()
    df['sentiment'] = df['merged_text'].apply(lambda x: sid.polarity_scores(x)['compound'])
    return df

def analyze_relationships(theta, sentiment, X, covar_names):
    K = theta.shape[1]
    results = {}
    for k in range(K):
        # Topic prevalence vs covariates
        model_p = sm.OLS(theta[:, k], X).fit()
        # Sentiment vs covariates
        model_s = sm.OLS(sentiment, X).fit()
        results[f'Topic_{k+1}'] = {
            'prevalence_coeffs': model_p.params.tolist(),
            'prevalence_se': model_p.bse.tolist(),
            'sentiment_coeffs': model_s.params.tolist(),
            'sentiment_se': model_s.bse.tolist(),
        }
    return results
