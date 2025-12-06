import re
import ast
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

# NLTK resources are downloaded lazily inside functions to avoid import-time failures

def _ensure_nltk():
    try:
        import nltk
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except Exception:
        import nltk
        nltk.download('stopwords')
        nltk.download('wordnet')


def merge_pos_neg(row: pd.Series) -> str:
    pos = row.get('Positive_Review', '') or ''
    neg = row.get('Negative_Review', '') or ''
    # Remove placeholders
    if isinstance(pos, str) and pos.strip().lower() in {'no positive', 'no positive '}: 
        pos = ''
    if isinstance(neg, str) and neg.strip().lower() in {'no negative', 'no negative '}: 
        neg = ''
    # Keep POS/NEG markers
    merged = f"POS: {pos} NEG: {neg}"
    return merged


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    # lowercasing
    s = text.lower()
    # remove urls
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    # remove numbers
    s = re.sub(r'\d+', ' ', s)
    # keep POS and NEG tags literal, remove other punctuation
    # Replace POS: and NEG: temporarily with tokens to preserve
    s = s.replace('pos:', ' POSTAG ').replace('neg:', ' NEGTAG ')
    s = re.sub(r"[^a-z\s]", ' ', s)
    # restore tags
    s = s.replace('postag', 'POS').replace('negtag', 'NEG')
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def tokenize_and_lemmatize(text: str, use_stem: bool = False) -> List[str]:
    _ensure_nltk()
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.stem import PorterStemmer

    stops = set(stopwords.words('english'))
    # Keep POS and NEG tags
    tokens = [t for t in re.split(r'\s+', text) if t]
    out = []
    wnl = WordNetLemmatizer()
    stemmer = PorterStemmer() if use_stem else None
    for tok in tokens:
        if tok in {'POS', 'NEG'}:
            out.append(tok)
            continue
        if tok in stops:
            continue
        if len(tok) <= 2:
            continue
        if use_stem:
            tok2 = stemmer.stem(tok)
        else:
            tok2 = wnl.lemmatize(tok)
        out.append(tok2)
    return out


def parse_tags_trip_type(tags_val: str) -> Dict[str, int]:
    """Parse Tags column into one-hot trip types.
    Expected tags like: "[' Leisure trip ', ' Couple ', ...]"
    """
    cats = {
        'Leisure trip': 0,
        'Business trip': 0,
        'Solo traveler': 0,
        'Family with young children': 0,
        'Group': 0,
        'Travelers with friends': 0,
        'With a pet': 0,
    }
    if not isinstance(tags_val, str) or not tags_val:
        return cats
    try:
        # Some rows may not be valid python lists; try gentle parsing
        lst = ast.literal_eval(tags_val)
        if not isinstance(lst, list):
            return cats
        for item in lst:
            if not isinstance(item, str):
                continue
            s = item.strip().strip("'\"")
            # normalize spaces
            s = re.sub(r'\s+', ' ', s).strip()
            if s in cats:
                cats[s] = 1
        return cats
    except Exception:
        # Fallback: simple splitting by comma
        for item in tags_val.split(','):
            s = item.strip().strip("'\"")
            s = re.sub(r'\s+', ' ', s).strip()
            if s in cats:
                cats[s] = 1
        return cats


def safe_parse_days_since(val: str) -> float:
    """Parse days_since_review string like '105 day', '0 days'. Return np.nan if fail."""
    if not isinstance(val, str):
        return np.nan
    m = re.search(r'(\d+)', val)
    if m:
        return float(m.group(1))
    return np.nan


def build_corpus_and_covariates(df: pd.DataFrame, use_stem: bool = False) -> Tuple[pd.DataFrame, List[List[str]], List[str], np.ndarray]:
    """
    Returns:
      df2: with merged text and covariates columns
      token_lists: list of tokens per doc
      vocab: list of unique vocab terms
      X: covariate matrix with intercept
    """
    # merge text
    df2 = df.copy()
    df2['merged_text'] = df2.apply(merge_pos_neg, axis=1)
    df2['clean_text'] = df2['merged_text'].map(clean_text)
    # tokenize
    token_lists = df2['clean_text'].map(lambda s: tokenize_and_lemmatize(s, use_stem=use_stem)).tolist()

    # Build covariates
    # Reviewer_Score z-score
    rs = df2['Reviewer_Score'].astype(float)
    rs_mean = rs.mean()
    rs_std = rs.std(ddof=0) if rs.std(ddof=0) > 0 else 1.0
    df2['Reviewer_Score_z'] = (rs - rs_mean) / rs_std

    # Days since review numeric
    df2['days_since_num'] = df2['days_since_review'].map(safe_parse_days_since)
    ds = df2['days_since_num']
    ds_mean = np.nanmean(ds)
    ds_std = np.nanstd(ds) if np.nanstd(ds) > 0 else 1.0
    df2['days_since_z'] = (ds - ds_mean) / ds_std

    # Optional reviewer activity z
    if 'Total_Number_of_Reviews_Reviewer_Has_Given' in df2.columns:
        ra = df2['Total_Number_of_Reviews_Reviewer_Has_Given'].astype(float)
        ra_mean = ra.mean()
        ra_std = ra.std(ddof=0) if ra.std(ddof=0) > 0 else 1.0
        df2['reviewer_activity_z'] = (ra - ra_mean) / ra_std
    else:
        df2['reviewer_activity_z'] = 0.0

    # Trip type one-hot
    trip_cols = ['Leisure trip','Business trip','Solo traveler','Family with young children','Group','Travelers with friends','With a pet']
    for c in trip_cols:
        df2[f'trip_{c.replace(" ", "_")}'] = 0
    parsed = df2['Tags'].map(parse_tags_trip_type).tolist()
    for i, d in enumerate(parsed):
        for k, v in d.items():
            df2.loc[i, f'trip_{k.replace(" ", "_")}'] = v

    # Construct X (intercept + covariates)
    covar_cols = ['Reviewer_Score_z','days_since_z','reviewer_activity_z'] + [f'trip_{c.replace(" ", "_")}' for c in trip_cols]
    X_raw = df2[covar_cols].fillna(0.0).values
    intercept = np.ones((X_raw.shape[0], 1))
    X = np.hstack([intercept, X_raw])

    # Build vocabulary from tokens, with min frequency filtering
    from collections import Counter
    all_tokens = [tok for tl in token_lists for tok in tl if tok not in {'POS','NEG'}]
    counts = Counter(all_tokens)
    vocab = sorted([w for w, c in counts.items() if c >= 2])

    return df2, token_lists, vocab, X


def build_doc_term_matrix(token_lists: List[List[str]], vocab: List[str]) -> np.ndarray:
    idx = {w: i for i, w in enumerate(vocab)}
    D = len(token_lists)
    V = len(vocab)
    dtm = np.zeros((D, V), dtype=np.int32)
    for d, tl in enumerate(token_lists):
        for tok in tl:
            if tok in {'POS','NEG'}:
                continue
            i = idx.get(tok)
            if i is not None:
                dtm[d, i] += 1
    return dtm
