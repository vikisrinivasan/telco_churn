# -*- coding: utf-8 -*-
"""Step 5 of the pipeline: turn a scored feature table into the quarterly call list.

Scores every customer in features.pkl with the combined model, assigns a microsegment and
marks the top 20% (the retention team's capacity). On the case-study data this scores
customers the model was trained on, so use the notebook's out-of-fold numbers for
evaluation; in production the input is the next quarter's T0 snapshot.

Output: reports/call_list.csv
"""
import argparse
import os
import pickle

import pandas as pd

from src.data.make_dataset import INTERIM_DIR, ROOT_DIR
from src.features.build_features import PROCESSED_DIR
from src.models.churn_model import CAPACITY
from src.models.microsegments import assign_segments
from src.models.train_model import feature_sets


def main(args):
    features = pd.read_pickle(os.path.join(args.processed_data_path, "features.pkl"))
    customers = pd.read_pickle(os.path.join(args.interim_data_path, "customers.pkl")).set_index("CustomerID")
    with open(os.path.join(args.model_logging, "singtel", "combined_model.pkl"), "rb") as f:
        model = pickle.load(f)

    X = feature_sets(features)["combined"]
    out = customers[["Contract", "Internet Service", "Tenure Months", "Monthly Charges"]].join(
        X[["protection_addons", "exit_intent_calls", "complaint_calls"]])
    out["churn_score"] = model.predict_proba(X)[:, 1]
    out["segment"] = assign_segments(out)
    out = out.sort_values("churn_score", ascending=False)
    out["rank"] = range(1, len(out) + 1)
    out["call_this_quarter"] = out["rank"] <= round(CAPACITY * len(out))
    out.to_csv(os.path.join(ROOT_DIR, "reports", "call_list.csv"))
    print(out[out.call_this_quarter].segment.value_counts())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score customers and write the call list.')
    parser.add_argument('--processed_data_path', type=str, default=PROCESSED_DIR, help='Path for processed data')
    parser.add_argument('--interim_data_path', type=str, default=INTERIM_DIR, help='Path for interim data')
    parser.add_argument('--model_logging', type=str, default=os.path.join(ROOT_DIR, "models"),
                        help='Path for saved models')
    main(parser.parse_args())
