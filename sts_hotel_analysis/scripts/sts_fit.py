import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from utils.text_preprocess import build_corpus_and_covariates, build_doc_term_matrix
from utils.anchor_init import nmf_initialize, theta_to_alpha_p
from utils.sts_model import STSModel
from utils.evaluation import heldout_log_likelihood, topic_coherence_umass
from utils.plotting import plot_elbo_curve, plot_score_relationships, plot_triptype_diffs, plot_time_trend

# Paths
DATA_PATH = 'sts_hotel_analysis/data/Hotel_Reviews.csv'
OUT_DIR = 'sts_hotel_analysis/outputs'

os.makedirs(OUT_DIR, exist_ok=True)


def run_em(model: STSModel, max_iters: int = 40, tol: float = 1e-5) -> List[float]:
    elbos = []
    prev = None
    for it in range(max_iters):
        model.e_step(maxiter=200)
        stats_gs = model.m_step_update_gamma_sigma()
        stats_k = model.m_step_update_kappa()
        elbo = model.compute_elbo()
        elbos.append(elbo)
        if prev is not None:
            rel = abs(elbo - prev) / (abs(prev) + 1e-9)
            if rel < tol:
                break
        prev = elbo
    return elbos


def select_K(df2: pd.DataFrame, token_lists: List[List[str]], vocab: List[str], X: np.ndarray,
              grid: List[int]) -> Dict:
    results = []
    dtm = build_doc_term_matrix(token_lists, vocab)
    for K in grid:
        theta_init, kappa_t_init, m_v = nmf_initialize(dtm, K)
        alpha_p_init = theta_to_alpha_p(theta_init)
        # sentiment init from reviewer score z
        a_s_init = np.tile(df2['Reviewer_Score_z'].values.reshape(-1, 1), (1, K))
        kappa_s_init = np.zeros((K, len(vocab)))
        model = STSModel(K, vocab, X, dtm, m_v, kappa_t_init, kappa_s_init, alpha_p_init, a_s_init)
        elbos = run_em(model, max_iters=25, tol=1e-4)
        held = heldout_log_likelihood(model)
        coh = topic_coherence_umass(model)
        results.append({'K': K, 'final_elbo': elbos[-1], 'heldout_ll': held, 'coherence_umass': coh})
    # Choose K by maximizing heldout_ll first; tie-breaker by coherence
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, 'model_selection.csv'), index=False)
    # Rank
    res_df['rank'] = res_df['heldout_ll'].rank(ascending=False, method='min') + res_df['coherence_umass'].rank(ascending=False, method='min')
    best = res_df.sort_values(['rank','heldout_ll','coherence_umass'], ascending=[True, False, False]).iloc[0]
    return {'best_K': int(best['K']), 'results': results}


def extract_keywords(model: STSModel, topn: int = 12) -> Dict:
    # baseline (a_s=0), low (min a_s), high (max a_s)
    a_s_vals = model.alpha_s
    low = float(np.percentile(a_s_vals, 10))
    high = float(np.percentile(a_s_vals, 90))
    base = model.top_words(topn=topn, a_s_level=0.0)
    loww = []
    highw = []
    for k in range(model.K):
        loww.append(model.top_words(topn=topn, a_s_level=low)[k])
        highw.append(model.top_words(topn=topn, a_s_level=high)[k])
    out = {'baseline': base, 'low_sentiment': loww, 'high_sentiment': highw,
           'low_level': low, 'high_level': high}
    return out


def extract_doc_examples(df2: pd.DataFrame, model: STSModel, n_per_topic: int = 3) -> Dict:
    examples = {}
    # Compute theta per doc
    theta_all = np.array([model.softmax(model.alpha_p[d, :]) for d in range(model.D)])
    for k in range(model.K):
        idx = np.argsort(theta_all[:, k])[::-1][:n_per_topic]
        ex = []
        for d in idx:
            text = df2.loc[d, 'merged_text']
            # Shorten
            snippet = text[:350]
            ex.append({'doc_index': int(d), 'Reviewer_Score': float(df2.loc[d, 'Reviewer_Score']), 'snippet': snippet})
        examples[f'Topic_{k+1}'] = ex
    return examples


def main():
    df = pd.read_csv(DATA_PATH)

    # Build corpus and covariates
    df2, token_lists, vocab, X = build_corpus_and_covariates(df, use_stem=False)
    # K selection
    grid = [5,6,7,8,9,10]
    sel = select_K(df2, token_lists, vocab, X, grid)
    K = sel['best_K']
    # Final fit
    dtm = build_doc_term_matrix(token_lists, vocab)
    theta_init, kappa_t_init, m_v = nmf_initialize(dtm, K)
    alpha_p_init = theta_to_alpha_p(theta_init)
    a_s_init = np.tile(df2['Reviewer_Score_z'].values.reshape(-1, 1), (1, K))
    kappa_s_init = np.zeros((K, len(vocab)))
    model = STSModel(K, vocab, X, dtm, m_v, kappa_t_init, kappa_s_init, alpha_p_init, a_s_init)
    elbos = run_em(model, max_iters=60, tol=1e-5)
    # Save ELBO curve
    plot_elbo_curve(elbos, os.path.join(OUT_DIR, 'elbo_curve.png'))
    # Regression coefficients outputs
    stats = model.m_step_update_gamma_sigma()
    Gamma = stats['Gamma']  # (P x 2K)
    se = stats['Gamma_se']
    # Build labeled DataFrame
    P = X.shape[1]
    covar_names = ['intercept','Reviewer_Score_z','days_since_z','reviewer_activity_z',
                   'trip_Leisure_trip','trip_Business_trip','trip_Solo_traveler','trip_Family_with_young_children',
                   'trip_Group','trip_Travelers_with_friends','trip_With_a_pet']
    cols = []
    for k in range(K):
        cols.append(f'g_p_T{k+1}')
    for k in range(K):
        cols.append(f'g_s_T{k+1}')
    df_gamma = pd.DataFrame(Gamma, index=covar_names, columns=cols)
    df_gamma_se = pd.DataFrame(se, index=covar_names, columns=cols)
    df_gamma.to_csv(os.path.join(OUT_DIR, 'gamma_coeffs.csv'))
    df_gamma_se.to_csv(os.path.join(OUT_DIR, 'gamma_coeffs_se.csv'))
    # Keywords
    kw = extract_keywords(model, topn=12)
    with open(os.path.join(OUT_DIR, 'topic_keywords.json'), 'w', encoding='utf-8') as f:
        json.dump(kw, f, ensure_ascii=False, indent=2)
    # Doc examples
    ex = extract_doc_examples(df2, model, n_per_topic=3)
    with open(os.path.join(OUT_DIR, 'doc_examples.json'), 'w', encoding='utf-8') as f:
        json.dump(ex, f, ensure_ascii=False, indent=2)
    # Relationship plots
    theta_all = np.array([model.softmax(model.alpha_p[d, :]) for d in range(model.D)])
    plot_score_relationships(theta_all, model.alpha_s, df2['Reviewer_Score_z'].values, os.path.join(OUT_DIR, 'score_relationships.png'))
    trip_cols = ['Leisure trip','Business trip','Solo traveler','Family with young children','Group','Travelers with friends','With a pet']
    plot_triptype_diffs(theta_all, model.alpha_s, df2, trip_cols, os.path.join(OUT_DIR, 'triptype_diffs.png'))
    plot_time_trend(theta_all, model.alpha_s, df2['days_since_z'].values, os.path.join(OUT_DIR, 'time_trend.png'))
    # Save selection summary
    with open(os.path.join(OUT_DIR, 'model_selection_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(sel, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
