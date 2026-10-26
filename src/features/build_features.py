# -*- coding: utf-8 -*-
"""Step 3 of the pipeline: interim tables -> one customer-level feature table.

Inputs (data/interim):  customers.pkl, calls.pkl
Outputs (data/processed):
    features.pkl      tabular + call-sequence + text_risk features, plus `churn` and `split`
    text_scorer.pkl   TF-IDF text model fitted on training customers (for scoring new data)

The train/test split is made here because the text score must be fitted on training
customers only: training rows get out-of-fold scores, test rows get the train-fitted model.
"""
import argparse
import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.make_dataset import INTERIM_DIR, ROOT_DIR
from src.features.tabular_features import build_tabular
from src.features.transcript_features import (TextRiskScorer, out_of_fold_text_risk,
                                              prepare_call_text, sequence_features)

SEED = 270
TEST_SIZE = 0.25
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")


def make_split(customers: pd.DataFrame, seed: int = SEED):
    """One stratified split reused by every model, so comparisons are like for like."""
    y = customers.set_index("CustomerID")["churn"]
    train_ids, test_ids = train_test_split(y.index, test_size=TEST_SIZE, stratify=y, random_state=seed)
    return train_ids, test_ids


def build_features(customers: pd.DataFrame, calls: pd.DataFrame):
    """Return (features, text_scorer). `features` is indexed by CustomerID."""
    y = customers.set_index("CustomerID")["churn"]
    train_ids, test_ids = make_split(customers)
    calls = prepare_call_text(calls)

    X = build_tabular(customers).join(sequence_features(calls, y.index))
    scorer = TextRiskScorer().fit(calls, list(train_ids), y.loc[train_ids])
    text = pd.concat([out_of_fold_text_risk(calls, list(train_ids), y.loc[train_ids]),
                      scorer.score(calls, list(test_ids))])
    X["text_risk"] = text.reindex(X.index)
    X["churn"] = y
    X["split"] = "train"
    X.loc[test_ids, "split"] = "test"
    return X, scorer


def main(args):
    os.makedirs(args.processed_data_path, exist_ok=True)
    customers = pd.read_pickle(os.path.join(args.interim_data_path, "customers.pkl"))
    calls = pd.read_pickle(os.path.join(args.interim_data_path, "calls.pkl"))
    features, scorer = build_features(customers, calls)
    features.to_pickle(os.path.join(args.processed_data_path, "features.pkl"))
    with open(os.path.join(args.processed_data_path, "text_scorer.pkl"), "wb") as f:
        pickle.dump(scorer, f)
    print(f"features {features.shape} | train {int((features.split == 'train').sum()):,} "
          f"| test {int((features.split == 'test').sum()):,} | callers {int(features.has_called.sum()):,}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build customer-level features.')
    parser.add_argument('--interim_data_path', type=str, default=INTERIM_DIR, help='Path for interim data')
    parser.add_argument('--processed_data_path', type=str, default=PROCESSED_DIR, help='Output path for processed data')
    main(parser.parse_args())
