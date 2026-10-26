"""Models and capacity-aware evaluation for the Singtel churn case.

The retention team can contact ~20% of the base per quarter, so every model is judged
on the customers it would put in that 20%: precision, recall and lift at k = 20%.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAPACITY = 0.20
SEED = 270


# ---------------------------------------------------------------- metrics
def at_capacity(y_true, score, k=CAPACITY):
    """Precision, recall and lift among the top-k share of customers by score."""
    y = np.asarray(y_true)
    n_top = int(round(k * len(y)))
    top = np.argsort(-np.asarray(score), kind="stable")[:n_top]
    caught = y[top].sum()
    precision = caught / n_top
    return {"precision@20": precision, "recall@20": caught / y.sum(),
            "lift@20": precision / y.mean(), "churners_caught": int(caught), "contacted": n_top}


def evaluate(y_true, score, threshold=0.5):
    out = at_capacity(y_true, score)
    out.update({"pr_auc": average_precision_score(y_true, score),
                "roc_auc": roc_auc_score(y_true, score),
                "accuracy@0.5": accuracy_score(y_true, np.asarray(score) >= threshold)})
    return out


def gain_curve(y_true, score, steps=100):
    """Share of churners captured as the contact list grows (cumulative gains)."""
    y = np.asarray(y_true)[np.argsort(-np.asarray(score), kind="stable")]
    cum = np.cumsum(y) / y.sum()
    xs = np.linspace(0, 1, steps + 1)
    idx = np.clip((xs * len(y)).astype(int) - 1, 0, len(y) - 1)
    return pd.DataFrame({"share_contacted": xs, "share_churners": np.where(xs == 0, 0, cum[idx])})


def bootstrap_delta(y_true, score_a, score_b, metric="recall@20", n=1000, seed=SEED):
    """Paired bootstrap of metric(b) - metric(a) on the same customers."""
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y_true, score_a, score_b))
    deltas, base = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        ma, mb = at_capacity(y[i], a[i])[metric], at_capacity(y[i], b[i])[metric]
        deltas.append(mb - ma)
        base.append(ma)
    deltas = np.array(deltas)
    return {"delta": at_capacity(y, b)[metric] - at_capacity(y, a)[metric],
            "ci_low": np.percentile(deltas, 2.5), "ci_high": np.percentile(deltas, 97.5),
            "p_b_better": float((deltas > 0).mean())}


def bootstrap_ci(y_true, score, metric="recall@20", n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y_true), np.asarray(score)
    vals = [at_capacity(y[i], s[i])[metric] for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


# ---------------------------------------------------------------- models
def logistic_pipeline(X, class_weight="balanced", C=0.5):
    cats = X.select_dtypes("category").columns.tolist()
    nums = [c for c in X.columns if c not in cats]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cats),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), nums),
    ])
    return Pipeline([("pre", pre),
                     ("lr", LogisticRegression(C=C, class_weight=class_weight, max_iter=5000))])


def lgbm_model(scale_pos_weight=None, seed=SEED):
    return lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight, random_state=seed, verbose=-1)


def cross_validate(make_model, X, y, n_splits=5, seed=SEED, fit_kwargs=None):
    """Out-of-fold scores and per-fold capacity metrics on the training set."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = pd.Series(np.nan, index=X.index)
    folds = []
    for fit_idx, val_idx in skf.split(X, y):
        model = make_model()
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx], **(fit_kwargs or {}))
        oof.iloc[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        folds.append(evaluate(y.iloc[val_idx], oof.iloc[val_idx]))
    return oof, pd.DataFrame(folds)


def score_table(customers, ids, scores, label="score"):
    """The brief's readout: each test customer, predicted score, outcome and recorded reason."""
    df = customers.set_index("CustomerID").loc[ids, ["churn", "Churn Reason", "Contract",
                                                      "Tenure Months", "Monthly Charges"]]
    df = df.assign(**{label: np.asarray(scores)}).sort_values(label, ascending=False)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df["in_top_20pct"] = df["rank"] <= round(CAPACITY * len(df))
    df["Churn Reason"] = df["Churn Reason"].replace("", "-")
    return df.reset_index()
