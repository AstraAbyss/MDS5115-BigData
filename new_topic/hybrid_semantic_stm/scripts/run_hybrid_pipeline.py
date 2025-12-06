#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Semantic STM Pipeline
- Merge hotel reviews with BERTopic doc topics
- Feature engineering from Tags
- Topic exposure regression (logit of topic_prob) with robust SE
- Sentiment proxy regression (Reviewer_Score ~ topic probs + covariates)
- Propensity Score Matching (Business vs Leisure) ATT on Reviewer_Score
- Compare with original STS coefficients
Outputs:
  experiments/hybrid_semantic_stm/outputs/prevalence_regression_results.csv
  experiments/hybrid_semantic_stm/outputs/sentiment_regression_results.csv
  experiments/hybrid_semantic_stm/outputs/psm_results.json
  experiments/hybrid_semantic_stm/outputs/psm_balance.csv
  experiments/hybrid_semantic_stm/outputs/comparison_vs_sts.csv
"""
import os
import re
import json
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

BASE_DIR = "experiments/hybrid_semantic_stm"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_HOTEL = "sts_hotel_sample_1k/data/Hotel_Reviews_1k_sample.csv"
DATA_BERTOPIC = "experiments/embedding_topics/outputs/bertopic_doc_topics.csv"
STS_PREV = "sts_hotel_sample_1k/outputs/coeffs_prevalence_sts.json"
STS_SENT = "sts_hotel_sample_1k/outputs/coeffs_sentiment_sts.json"

TAG_PATTERNS = {
    "is_leisure": re.compile(r"\bLeisure\s+trip\b", re.I),
    "is_business": re.compile(r"\bBusiness\s+trip\b", re.I),
    "is_couple": re.compile(r"\bCouple\b", re.I),
    "is_solo": re.compile(r"\bSolo\s+traveler\b", re.I),
}
STAYED_NIGHTS_RE = re.compile(r"Stayed\s+(\d+)\s+nights?", re.I)


def parse_tags_to_covariates(tags_str: str) -> Dict[str, float]:
    """Parse Tags column to binary covariates and stayed_nights.
    tags_str looks like "[' Leisure trip ', ' Couple ', ' Stayed 3 nights ', ...]"
    """
    cov = {k: 0.0 for k in TAG_PATTERNS.keys()}
    stayed_nights = np.nan
    if not isinstance(tags_str, str):
        cov["stayed_nights"] = stayed_nights
        return cov
    s = tags_str
    for k, pat in TAG_PATTERNS.items():
        cov[k] = 1.0 if pat.search(s) else 0.0
    m = STAYED_NIGHTS_RE.search(s)
    if m:
        try:
            stayed_nights = float(m.group(1))
        except Exception:
            stayed_nights = np.nan
    cov["stayed_nights"] = stayed_nights
    return cov


def load_and_merge() -> pd.DataFrame:
    df = pd.read_csv(DATA_HOTEL)
    dt = pd.read_csv(DATA_BERTOPIC)
    # drop noise topics
    dt = dt[dt["topic"].astype(float) != -1]
    # ensure types
    df["doc_id"] = df["doc_id"].astype(int)
    dt["doc_id"] = dt["doc_id"].astype(int)
    # merge
    m = pd.merge(df, dt[["doc_id", "topic", "topic_prob", "sent_score"]], on="doc_id", how="inner")
    # parse tags
    cov_parsed = m["Tags"].apply(parse_tags_to_covariates).apply(pd.Series)
    m = pd.concat([m, cov_parsed], axis=1)
    # impute stayed_nights (median)
    if m["stayed_nights"].isna().any():
        med = m["stayed_nights"].median()
        m["stayed_nights"] = m["stayed_nights"].fillna(med)
    # rename covariates
    m = m.rename(columns={
        "Reviewer_Score": "reviewer_score",
        "Total_Number_of_Reviews_Reviewer_Has_Given": "reviewer_reviews",
        "month_idx": "month_idx",
        "days_num": "days_num"
    })
    # topic vector: create columns for each topic present
    topics = sorted(m["topic"].dropna().astype(int).unique())
    for t in topics:
        col = f"topic_prob_{t}"
        m[col] = 0.0
    # fill assigned topic probability
    for idx, row in m.iterrows():
        t = int(row["topic"])
        m.at[idx, f"topic_prob_{t}"] = float(row["topic_prob"]) if not pd.isna(row["topic_prob"]) else 0.0
    # boundary shrink/clip helper
    return m


def boundary_clip_logit(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y = np.clip(y, eps, 1 - eps)
    return np.log(y / (1 - y))


def fit_topic_exposure_regressions(m: pd.DataFrame, topics: List[int]) -> pd.DataFrame:
    rows = []
    Xvars = [
        "reviewer_score", "month_idx", "days_num", "reviewer_reviews",
        "is_leisure", "is_business", "is_couple", "is_solo", "stayed_nights"
    ]
    # Drop rows with NA in key Xvars
    m_clean = m.copy()
    for v in Xvars:
        m_clean = m_clean[~m_clean[v].isna()]
    for t in topics:
        y_raw = m_clean[f"topic_prob_{t}"].values.astype(float)
        y_logit = boundary_clip_logit(y_raw)
        X = m_clean[Xvars].astype(float)
        X = sm.add_constant(X)
        model = sm.OLS(y_logit, X)
        res = model.fit(cov_type='HC3')
        r2 = float(res.rsquared)
        # collect coef, se, pval
        for var, coef, se, pval in zip(res.params.index, res.params.values, res.bse.values, res.pvalues.values):
            rows.append({
                "topic_id": t,
                "var": var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval),
                "r2_topic": r2
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "prevalence_regression_results.csv"), index=False)
    return out


def fit_sentiment_regression(m: pd.DataFrame, topics: List[int]) -> pd.DataFrame:
    rows = []
    y = m["reviewer_score"].astype(float)
    topic_cols = [f"topic_prob_{t}" for t in topics]
    Xvars = topic_cols + ["month_idx", "days_num", "reviewer_reviews", "is_leisure", "is_business", "is_couple", "is_solo", "stayed_nights"]
    m_clean = m.copy()
    for v in Xvars:
        m_clean = m_clean[~m_clean[v].isna()]
    X = m_clean[Xvars].astype(float)
    X = sm.add_constant(X)
    model = sm.OLS(m_clean["reviewer_score"].values.astype(float), X)
    res = model.fit(cov_type='HC3')
    r2 = float(res.rsquared)
    for var, coef, se, pval in zip(res.params.index, res.params.values, res.bse.values, res.pvalues.values):
        rows.append({
            "var": var,
            "coef": float(coef),
            "se": float(se),
            "pval": float(pval),
            "r2": r2
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "sentiment_regression_results.csv"), index=False)
    return out


def psm_match_and_att(m: pd.DataFrame, topics: List[int], caliper: float = 0.1) -> Tuple[Dict, pd.DataFrame]:
    # Treatment: Business trip (1 if is_business==1)
    df = m.copy()
    df["treat"] = (df["is_business"] == 1.0).astype(int)
    # Controls: Leisure trip (is_leisure==1)
    # Features for propensity: topic probs + month_idx + days_num + reviewer_reviews + stayed_nights
    feat_cols = [f"topic_prob_{t}" for t in topics] + ["month_idx", "days_num", "reviewer_reviews", "stayed_nights"]
    # Keep rows with valid treatment definition (either business or leisure); drop others
    mask_valid = (df["is_business"].isin([0.0, 1.0])) & (df["is_leisure"].isin([0.0, 1.0]))
    df = df[mask_valid]
    # Impute any NaNs in features with median
    for c in feat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    X = df[feat_cols].values.astype(float)
    y = df["treat"].values.astype(int)
    # Standardize features for logistic stability
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    logit = LogisticRegression(max_iter=1000, solver='lbfgs')
    logit.fit(Xs, y)
    ps = logit.predict_proba(Xs)[:, 1]
    df["propensity"] = ps
    # Nearest neighbor matching 1:1 without replacement, caliper on propensity
    treated = df[df["treat"] == 1].copy()
    control = df[df["treat"] == 0].copy()
    dists = pairwise_distances(treated[["propensity"]].values, control[["propensity"]].values, metric='euclidean')
    matched_pairs = []
    used_controls = set()
    for i in range(dists.shape[0]):
        # find nearest control within caliper
        j_sorted = np.argsort(dists[i, :])
        chosen = None
        for j in j_sorted:
            if j in used_controls:
                continue
            if dists[i, j] <= caliper:
                chosen = j
                break
        if chosen is not None:
            used_controls.add(chosen)
            matched_pairs.append((treated.index[i], control.index[chosen]))
    # Compute ATT
    diffs = []
    for ti, cj in matched_pairs:
        diffs.append(df.loc[ti, "reviewer_score"] - df.loc[cj, "reviewer_score"])
    att = float(np.mean(diffs)) if diffs else float('nan')
    se = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else float('nan')
    ci_low = att - 1.96 * se if not math.isnan(se) else float('nan')
    ci_high = att + 1.96 * se if not math.isnan(se) else float('nan')
    # Balance check: standardized differences before and after
    def std_diff(x_t, x_c):
        m1, m0 = np.mean(x_t), np.mean(x_c)
        s1, s0 = np.std(x_t, ddof=1), np.std(x_c, ddof=1)
        sp = math.sqrt((s1**2 + s0**2) / 2.0)
        return float((m1 - m0) / sp) if sp > 0 else 0.0
    bal_rows = []
    # before matching
    X_t = df[df["treat"] == 1]
    X_c = df[df["treat"] == 0]
    for c in feat_cols:
        before = std_diff(X_t[c].values, X_c[c].values)
        bal_rows.append({"var": c, "std_diff_before": before, "std_diff_after": np.nan})
    # after matching
    if matched_pairs:
        t_idx = [ti for ti, _ in matched_pairs]
        c_idx = [cj for _, cj in matched_pairs]
        X_tm = df.loc[t_idx]
        X_cm = df.loc[c_idx]
        for i, row in enumerate(bal_rows):
            c = row["var"]
            after = std_diff(X_tm[c].values, X_cm[c].values)
            bal_rows[i]["std_diff_after"] = after
    bal_df = pd.DataFrame(bal_rows)
    bal_df.to_csv(os.path.join(OUTPUT_DIR, "psm_balance.csv"), index=False)
    results = {
        "method": "1:1 nearest neighbor matching",
        "caliper": caliper,
        "n_treated": int((df["treat"] == 1).sum()),
        "n_control": int((df["treat"] == 0).sum()),
        "matched_pairs": int(len(matched_pairs)),
        "ATT": att if not math.isnan(att) else None,
        "CI_95_lower": ci_low if not math.isnan(ci_low) else None,
        "CI_95_upper": ci_high if not math.isnan(ci_high) else None
    }
    # JSON writing: use null not NaN
    with open(os.path.join(OUTPUT_DIR, "psm_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, separators=(",", ":"))
    return results, bal_df


def compare_with_sts(m: pd.DataFrame, topics: List[int]) -> pd.DataFrame:
    # Load our regressions
    preval = pd.read_csv(os.path.join(OUTPUT_DIR, "prevalence_regression_results.csv"))
    sent = pd.read_csv(os.path.join(OUTPUT_DIR, "sentiment_regression_results.csv"))
    # Load mapping if exists (sts_topic_id -> bertopic_topic_id)
    map_path = "experiments/embedding_topics/outputs/topic_mapping.csv"
    mapping = None
    if os.path.exists(map_path):
        mapping = pd.read_csv(map_path)
    # Load STS coeffs
    try:
        with open(STS_PREV, "r", encoding="utf-8") as f:
            sts_prev = json.load(f)
    except Exception:
        sts_prev = None
    try:
        with open(STS_SENT, "r", encoding="utf-8") as f:
            sts_sent = json.load(f)
    except Exception:
        sts_sent = None
    rows = []
    # Assumption: index 1 (second element) in STS coefs corresponds to Reviewer_Score (Rating) effect
    rating_idx = 1
    if sts_prev and "topics" in sts_prev:
        for t_entry in sts_prev["topics"]:
            sts_tid = t_entry.get("topic")
            coefs = t_entry.get("coefs", [])
            sts_rating_coef = coefs[rating_idx] if len(coefs) > rating_idx else None
            # find mapped bertopic topic id
            bert_tid = None
            match_score = None
            if mapping is not None:
                hit = mapping[mapping["sts_topic_id"] == sts_tid]
                if not hit.empty:
                    bert_tid = int(hit.iloc[0]["bertopic_topic_id"])
                    match_score = float(hit.iloc[0].get("match_score", np.nan))
            # our prevalence rating coefficient
            our_prev = preval[(preval["topic_id"] == bert_tid) & (preval["var"] == "reviewer_score")] if bert_tid is not None else pd.DataFrame()
            our_prev_coef = float(our_prev.iloc[0]["coef"]) if not our_prev.empty else None
            sign_agree = None
            if sts_rating_coef is not None and our_prev_coef is not None:
                sign_agree = "yes" if (np.sign(sts_rating_coef) == np.sign(our_prev_coef)) else "no"
            rows.append({
                "dimension": "prevalence",
                "topic_ref": f"STS_{sts_tid} -> BERT_{bert_tid}" if bert_tid is not None else f"STS_{sts_tid} -> BERT_?",
                "var": "reviewer_score",
                "sign_agree": sign_agree if sign_agree is not None else "NA",
                "significance_pattern": "NA",
                "notes": f"match_score={match_score}" if match_score is not None else "no mapping"
            })
    if sts_sent and "topics" in sts_sent:
        for t_entry in sts_sent["topics"]:
            sts_tid = t_entry.get("topic")
            coefs = t_entry.get("coefs", [])
            sts_rating_coef = coefs[rating_idx] if len(coefs) > rating_idx else None
            bert_tid = None
            match_score = None
            if mapping is not None:
                hit = mapping[mapping["sts_topic_id"] == sts_tid]
                if not hit.empty:
                    bert_tid = int(hit.iloc[0]["bertopic_topic_id"])
                    match_score = float(hit.iloc[0].get("match_score", np.nan))
            # our sentiment regression: coefficient of topic_prob_{bert_tid} on reviewer_score
            varname = f"topic_prob_{bert_tid}" if bert_tid is not None else None
            our_sent = sent[sent["var"] == varname] if varname is not None else pd.DataFrame()
            our_sent_coef = float(our_sent.iloc[0]["coef"]) if not our_sent.empty else None
            sign_agree = None
            # Here we compare: STS sentiment regression rating->topic sentiment vs our sentiment regression topic_prob->rating
            # Directions not directly comparable; we record direction with caution
            if sts_rating_coef is not None and our_sent_coef is not None:
                sign_agree = "partial" if np.sign(sts_rating_coef) == np.sign(our_sent_coef) else "diff"
            rows.append({
                "dimension": "sentiment",
                "topic_ref": f"STS_{sts_tid} -> BERT_{bert_tid}" if bert_tid is not None else f"STS_{sts_tid} -> BERT_?",
                "var": "rating_vs_topic_prob",
                "sign_agree": sign_agree if sign_agree is not None else "NA",
                "significance_pattern": "NA",
                "notes": f"match_score={match_score}; comparison uses direction only"
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "comparison_vs_sts.csv"), index=False)
    return out


def main():
    m = load_and_merge()
    topics = sorted(m["topic"].astype(int).unique())
    # 2) topic exposure regressions
    fit_topic_exposure_regressions(m, topics)
    # 3) sentiment regression
    fit_sentiment_regression(m, topics)
    # 4) PSM
    psm_match_and_att(m, topics, caliper=0.1)
    # 5) comparison
    compare_with_sts(m, topics)
    print("All outputs written to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
