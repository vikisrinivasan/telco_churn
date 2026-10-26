# -*- coding: utf-8 -*-
"""Step 4 of the pipeline: fit the tabular and combined models, evaluate at 20% capacity.

Inputs (data/processed):  features.pkl
Outputs:
    models/singtel/{tabular,combined}_model.pkl
    models/singtel/metrics.json                   test metrics for both models + bootstrap delta
    reports/test_scores_{tabular,combined}.csv    score vs recorded churn reason, ranked
"""
import argparse
import json
import os
import pickle

import pandas as pd

from src.data.make_dataset import INTERIM_DIR, ROOT_DIR
from src.features.build_features import PROCESSED_DIR
from src.models.churn_model import bootstrap_delta, evaluate, logistic_pipeline, score_table

SEQUENCE_AND_TEXT = ["has_called", "n_calls", "n_calls_90d", "n_inbound", "n_outbound", "n_failed_calls",
                     "days_since_last_call", "days_since_first_call", "max_severity", "last_severity",
                     "severity_slope", "exit_intent_calls", "last_call_exit_intent", "complaint_calls",
                     "unresolved_share", "last_call_unresolved", "repeat_contacts_7d", "competitor_mentions",
                     "min_csat", "mean_asr_confidence", "text_risk"]


def feature_sets(features: pd.DataFrame):
    inputs = features.drop(columns=["churn", "split"])
    return {"tabular": inputs.drop(columns=SEQUENCE_AND_TEXT), "combined": inputs}


def main(args):
    features = pd.read_pickle(os.path.join(args.processed_data_path, "features.pkl"))
    customers = pd.read_pickle(os.path.join(args.interim_data_path, "customers.pkl"))
    train, test = features.split == "train", features.split == "test"
    y = features["churn"]
    out_dir = os.path.join(args.model_logging, "singtel")
    report_dir = os.path.join(ROOT_DIR, "reports")
    os.makedirs(out_dir, exist_ok=True)

    scores, metrics = {}, {}
    for name, X in feature_sets(features).items():
        model = logistic_pipeline(X[train]).fit(X[train], y[train])
        scores[name] = model.predict_proba(X[test])[:, 1]
        metrics[name] = evaluate(y[test], scores[name])
        with open(os.path.join(out_dir, f"{name}_model.pkl"), "wb") as f:
            pickle.dump(model, f)
        score_table(customers, X[test].index, scores[name], "churn_score").to_csv(
            os.path.join(report_dir, f"test_scores_{name}.csv"), index=False)

    metrics["delta_recall@20"] = bootstrap_delta(y[test], scores["tabular"], scores["combined"])
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, default=float)
    print(pd.DataFrame({k: v for k, v in metrics.items() if k != "delta_recall@20"}).T
          [["recall@20", "precision@20", "lift@20", "pr_auc"]].round(3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate churn models.')
    parser.add_argument('--processed_data_path', type=str, default=PROCESSED_DIR, help='Path for processed data')
    parser.add_argument('--interim_data_path', type=str, default=INTERIM_DIR, help='Path for interim data')
    parser.add_argument('--model_logging', type=str, default=os.path.join(ROOT_DIR, "models"),
                        help='Output path for model data')
    main(parser.parse_args())
