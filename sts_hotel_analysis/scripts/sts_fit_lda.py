import json
import os
import numpy as np
import pandas as pd

from utils.text_preprocess import build_corpus_and_covariates, build_doc_term_matrix
from utils.lda_analysis import select_K_lda, get_sentiment, analyze_relationships
from utils.plotting import plot_score_relationships, plot_triptype_diffs, plot_time_trend

import nltk

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


# Paths
DATA_PATH = 'sts_hotel_analysis/data/Hotel_Reviews.csv'
OUT_DIR = 'sts_hotel_analysis/outputs'

os.makedirs(OUT_DIR, exist_ok=True)

def get_top_words(model, feature_names, n_top_words):
    top_words = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words[f'Topic_{topic_idx+1}'] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return top_words

def main():
    df = pd.read_csv(DATA_PATH)

    # Preprocessing
    df2, token_lists, vocab, X = build_corpus_and_covariates(df, use_stem=False)
    df2 = get_sentiment(df2)

    # K selection for LDA
    grid = [5, 6, 7, 8, 9, 10]
    selection_results = select_K_lda(token_lists, grid)
    best_K = selection_results['best_K']
    dtm = selection_results['dtm']
    vectorizer = selection_results['vectorizer']

    # Final LDA model
    lda = select_K_lda(token_lists, [best_K])['dtm']
    lda_model = LatentDirichletAllocation(n_components=best_K, max_iter=20, random_state=0,
                                   learning_method='online', learning_offset=50., n_jobs=-1)
    theta = lda_model.fit_transform(dtm)

    # Topic Keywords
    feature_names = vectorizer.get_feature_names_out()
    top_words = get_top_words(lda_model, feature_names, 12)
    with open(os.path.join(OUT_DIR, 'topic_keywords.json'), 'w', encoding='utf-8') as f:
        json.dump(top_words, f, ensure_ascii=False, indent=2)

    # Covariate Analysis
    covar_names = ['intercept', 'Reviewer_Score_z', 'days_since_z', 'reviewer_activity_z',
                   'trip_Leisure_trip', 'trip_Business_trip', 'trip_Solo_traveler', 'trip_Family_with_young_children',
                   'trip_Group', 'trip_Travelers_with_friends', 'trip_With_a_pet']
    
    # We will fake the alpha_s with sentiment scores for plotting
    alpha_s_fake = np.tile(df2['sentiment'].values.reshape(-1,1), (1, best_K))

    # Relationship plots
    plot_score_relationships(theta, alpha_s_fake, df2['Reviewer_Score_z'].values, os.path.join(OUT_DIR, 'score_relationships.png'))
    trip_cols = ['Leisure trip', 'Business trip', 'Solo traveler', 'Family with young children', 'Group', 'Travelers with friends', 'With a pet']
    plot_triptype_diffs(theta, alpha_s_fake, df2, trip_cols, os.path.join(OUT_DIR, 'triptype_diffs.png'))
    plot_time_trend(theta, alpha_s_fake, df2['days_since_z'].values, os.path.join(OUT_DIR, 'time_trend.png'))
    
    # Save regression coefficients for prevalence
    prevalence_coeffs = {}
    for k in range(best_K):
        model_p = sm.OLS(theta[:, k], X).fit()
        prevalence_coeffs[f'Topic_{k+1}'] = {
            'params': dict(zip(covar_names, model_p.params)),
            'se': dict(zip(covar_names, model_p.bse))
        }
    with open(os.path.join(OUT_DIR, 'prevalence_coeffs.json'), 'w', encoding='utf-8') as f:
        json.dump(prevalence_coeffs, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    from sklearn.decomposition import LatentDirichletAllocation
    import statsmodels.api as sm
    main()
