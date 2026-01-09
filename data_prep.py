#!/usr/bin/env python3
"""
data_prep_new.py

Robust data preparation for the phishing/fraud detector hackathon.

Features:
- Auto-detects common spam/phishing CSV formats (v1/v2 or text/label)
- Normalizes labels to 'benign', 'phishing', 'fraud' (adjust mapping as needed)
- Cleans message text (strip, remove control chars, collapse spaces)
- Optional class balancing (upsample minority classes)
- Optional train/val/test split and writes CSV outputs

Usage examples:
# Basic conversion from an uploaded path (adjust path for Windows)
python data_prep_new.py --in "/mnt/data/spam.csv" --out sample_data.csv

# Create train/val/test splits and balance classes
python data_prep_new.py --in "spam.csv" --out sample_data.csv --split 0.8 0.1 0.1 --balance --seed 42

# If your file uses different column names, specify them:
python data_prep_new.py --in "spam.csv" --text-col "v2" --label-col "v1"
"""
import argparse
import os
import re
import sys
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------- Utilities ----------

CONTROL_CHAR_RE = re.compile(r'[\r\n\t]+')
MULTISPACE_RE = re.compile(r'\s{2,}')

DEFAULT_LABEL_MAP = {
    # map common variants to our canonical labels
    'ham': 'benign',
    'spam': 'fraud',   # Kaggle uses spam for unsolicited messages; treat as fraud/scam
    'legit': 'benign',
    'phish': 'phishing',
    'phishing': 'phishing',
    'fraud': 'fraud',
    '0': 'benign',
    '1': 'fraud',
}


def clean_text(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    # remove control characters, collapse whitespace, strip
    s = CONTROL_CHAR_RE.sub(' ', s)
    s = MULTISPACE_RE.sub(' ', s)
    return s.strip()


def detect_columns(df: pd.DataFrame, text_col_arg: str = None, label_col_arg: str = None):
    cols = [c.lower() for c in df.columns]
    text_col = None
    label_col = None

    # if user provided explicit names, prefer them (case-insensitive)
    if text_col_arg:
        for c in df.columns:
            if c.lower() == text_col_arg.lower():
                text_col = c
                break
    if label_col_arg:
        for c in df.columns:
            if c.lower() == label_col_arg.lower():
                label_col = c
                break

    if text_col and label_col:
        return text_col, label_col

    # common Kaggle SMS spam format: v1 (label), v2 (text)
    if 'v2' in cols and 'v1' in cols:
        # pick actual column names from original df
        text_col = [c for c in df.columns if c.lower() == 'v2'][0]
        label_col = [c for c in df.columns if c.lower() == 'v1'][0]
        return text_col, label_col

    # look for likely text column
    for candidate in ['text', 'message', 'msg', 'body', 'content', 'v2']:
        for c in df.columns:
            if c.lower() == candidate:
                text_col = c
                break
        if text_col:
            break

    # look for likely label column
    for candidate in ['label', 'class', 'type', 'v1', 'spam']:
        for c in df.columns:
            if c.lower() == candidate:
                label_col = c
                break
        if label_col:
            break

    return text_col, label_col


def normalize_label(x: str, label_map=None):
    if pd.isna(x):
        return 'benign'
    s = str(x).strip().lower()
    if label_map and s in label_map:
        return label_map[s]
    # fallback heuristics
    if s in ['ham', 'legit', '0', 'normal']:
        return 'benign'
    if 'phish' in s:
        return 'phishing'
    if s in ['spam', 'scam', 'fraud', '1']:
        return 'fraud'
    return s or 'benign'


# ---------- Main pipeline ----------

def prepare_dataset(
    in_path,
    out_path,
    text_col_arg=None,
    label_col_arg=None,
    balance=False,
    split=None,
    seed=42,
    label_map=None,
    min_text_len=3
):
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # try common encodings
    tried = []
    for enc in ('utf-8', 'latin1', 'cp1252'):
        try:
            df = pd.read_csv(in_path, encoding=enc, low_memory=False)
            break
        except Exception as e:
            tried.append((enc, str(e)))
    else:
        raise RuntimeError(f"Failed to read {in_path}. Tried encodings: {tried}")

    # detect columns
    text_col, label_col = detect_columns(df, text_col_arg, label_col_arg)
    if text_col is None or label_col is None:
        raise SystemExit(f"Could not auto-detect text and label columns.\nColumns found: {list(df.columns)}\n"
                         f"Please re-run with --text-col and --label-col to specify them.")

    # keep only needed columns
    df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
    # clean text
    df['text'] = df['text'].map(clean_text)
    # drop very short / empty texts
    df = df[df['text'].str.len() >= min_text_len].copy()

    # normalize labels
    merged_label_map = dict(DEFAULT_LABEL_MAP)
    if label_map:
        merged_label_map.update(label_map)
    df['label'] = df['label'].map(lambda x: normalize_label(x, merged_label_map))

    # report distribution
    counts = Counter(df['label'].tolist())
    print("Initial class distribution:", counts)

    # Optional balancing (simple upsampling of minority classes)
    if balance:
        print("Balancing classes by upsampling minority classes...")
        max_n = max(counts.values())
        frames = []
        rng = pd.np.random.RandomState(seed) if hasattr(pd, 'np') else __import__('numpy').random.RandomState(seed)
        for lab, grp in df.groupby('label'):
            n = len(grp)
            if n < max_n:
                # sample with replacement
                sampled = grp.sample(max_n - n, replace=True, random_state=seed)
                frames.append(pd.concat([grp, sampled], axis=0))
            else:
                frames.append(grp)
        df = pd.concat(frames).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        counts = Counter(df['label'].tolist())
        print("Post-balance class distribution:", counts)

    # Output single CSV or split
    if split:
        if len(split) != 3 or not abs(sum(split) - 1.0) < 1e-6:
            raise ValueError("split must be three floats that sum to 1.0 (train, val, test)")
        train_frac, val_frac, test_frac = split
        # first split train vs temp
        train_df, temp_df = train_test_split(df, train_size=train_frac, random_state=seed, stratify=df['label'])
        # compute relative val share within temp
        val_rel = val_frac / (val_frac + test_frac)
        val_df, test_df = train_test_split(temp_df, train_size=val_rel, random_state=seed, stratify=temp_df['label'])
        # write files
        base, ext = os.path.splitext(out_path)
        train_path = f"{base}.train.csv"
        val_path = f"{base}.val.csv"
        test_path = f"{base}.test.csv"
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"Wrote train/val/test to: {train_path}, {val_path}, {test_path}")
        return train_df, val_df, test_df
    else:
        df.to_csv(out_path, index=False)
        print(f"Wrote unified dataset to {out_path}")
        return df


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Prepare dataset for phishing/fraud detection")
    p.add_argument('--in', dest='in_path', required=True, help='Input CSV path (e.g. spam.csv or /mnt/data/spam.csv)')
    p.add_argument('--out', dest='out_path', default='sample_data.csv', help='Output CSV path')
    p.add_argument('--text-col', dest='text_col', default=None, help='Name of the text column if autodetect fails')
    p.add_argument('--label-col', dest='label_col', default=None, help='Name of the label column if autodetect fails')
    p.add_argument('--balance', action='store_true', help='Upsample minority classes to match majority')
    p.add_argument('--split', nargs=3, type=float, metavar=('TRAIN','VAL','TEST'), help='Provide three floats summing to 1.0 to create splits')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    return p.parse_args()


def main():
    args = parse_args()
    split = tuple(args.split) if args.split else None
    prepare_dataset(
        in_path=args.in_path,
        out_path=args.out_path,
        text_col_arg=args.text_col,
        label_col_arg=args.label_col,
        balance=args.balance,
        split=split,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
