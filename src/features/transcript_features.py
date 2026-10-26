"""Turn each customer's pre-T0 contact history into features.

Two families:
  1. Sequence features from call metadata, in time order (how often, how recently,
     whether contacts are escalating, what the last call was about).
  2. A text score: TF-IDF over customer turns, recency-weighted per customer, fed to
     a regularised logistic regression. It is fitted only on training customers and
     produced out-of-fold for them, so the score never sees its own label.

Customers who never called get has_called = 0, zero counts, and NaN for "time since"
and text score. "Never called" is a real state, not missing data.
"""
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.data.make_dataset import T0

# ---------------------------------------------------------------- transcript text cleaning
FILLERS = r"\b(?:um+|uh+|erm+|mm+|hmm+|like|i mean|basically|you know|sort of|kind of)\b"
ASR_TAGS = r"\[(?:inaudible|unintelligible|pause|background noise|crosstalk|call disconnected|no audio captured)\]"
COMPETITORS = ["kestrel mobile", "orbit communications", "northgate telecom", "nexafone", "vantel"]


def customer_turns(transcript):
    """Keep only what the customer said; agent turns are scripted and carry the offer we made."""
    if not isinstance(transcript, str):
        return ""
    turns = [line.split(":", 1)[1] for line in transcript.split("\n")
             if line.startswith("Customer:") and ":" in line]
    return " ".join(turns)


def normalise_text(text):
    """Lower-case, mask competitor names and amounts, strip ASR tags and fillers."""
    text = text.lower()
    for name in COMPETITORS:
        text = text.replace(name, " competitorname ")
    text = re.sub(ASR_TAGS, " ", text)
    text = re.sub(r"\$\s?\d+(?:[.,]\d+)?", " moneyamount ", text)
    text = re.sub(r"\bref-\d+\b", " ", text)
    text = re.sub(FILLERS, " ", text)
    text = re.sub(r"[^a-z' ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def mentions_competitor(transcript):
    if not isinstance(transcript, str):
        return False
    low = transcript.lower()
    return any(name in low for name in COMPETITORS) or bool(
        re.search(r"(offer|quote|deal) from|competitor|cheaper elsewhere", low))


# Severity of the call reason, in the order a retention lead would worry about it.
SEVERITY = {
    "Service Enquiry": 0, "Add-on Purchase": 0, "Move / Relocation": 1, "Billing Enquiry": 1,
    "Plan Change Request": 1, "Technical Fault": 2, "Contract Enquiry": 2, "Billing Dispute": 2,
    "Complaint / Escalation": 3, "Retention Offer Discussion": 3,
    "Number Porting Enquiry": 4, "Termination Request": 4,
}
EXIT_INTENT = ["Termination Request", "Number Porting Enquiry"]
HALF_LIFE_DAYS = 45


def prepare_call_text(calls):
    calls = calls.copy()
    calls["customer_text"] = [normalise_text(customer_turns(t)) if not s else ""
                              for t, s in zip(calls["transcript"], calls["is_stub"])]
    calls["competitor"] = calls["transcript"].map(mentions_competitor)
    calls["days_before_t0"] = (T0 - calls["interaction_date"]).dt.days
    calls["recency_weight"] = 0.5 ** (calls["days_before_t0"] / HALF_LIFE_DAYS)
    calls["severity"] = calls["call_type"].map(SEVERITY)
    calls["unresolved"] = calls["resolution_status"].isin(["Unresolved", "Escalated", "Pending Callback"])
    return calls


def _slope(x, y):
    if len(x) < 2 or np.ptp(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def sequence_features(calls, index):
    """One row per customer in `index`. Calls must already be filtered to <= T0."""
    rows = {}
    for cid, g in calls.groupby("customer_id"):
        g = g.sort_values("interaction_date")
        inbound = g[g["channel"] == "Inbound Voice"]
        last = g.iloc[-1]
        rows[cid] = {
            "has_called": 1,
            "n_calls": len(g),
            "n_calls_90d": int((g["days_before_t0"] <= 90).sum()),
            "n_inbound": len(inbound),
            "n_outbound": int((g["channel"] == "Outbound Voice").sum()),
            "n_failed_calls": int(g["is_stub"].sum()),
            "days_since_last_call": int(last["days_before_t0"]),
            "days_since_first_call": int(g["days_before_t0"].max()),
            "max_severity": g["severity"].max(),
            "last_severity": last["severity"],
            # Escalation: are later calls more serious than earlier ones?
            "severity_slope": _slope(-g["days_before_t0"].values / 30.0, g["severity"].values),
            "exit_intent_calls": int(g["call_type"].isin(EXIT_INTENT).sum()),
            "last_call_exit_intent": int(last["call_type"] in EXIT_INTENT),
            "complaint_calls": int(g["call_type"].isin(["Complaint / Escalation", "Billing Dispute"]).sum()),
            "unresolved_share": g["unresolved"].mean(),
            "last_call_unresolved": int(last["unresolved"]),
            "repeat_contacts_7d": int(g["repeat_contact_7d"].sum()),
            "competitor_mentions": int(g["competitor"].sum()),
            "min_csat": g["csat"].min(),
            "mean_asr_confidence": g["asr_confidence"].mean(),
        }
    feats = pd.DataFrame.from_dict(rows, orient="index").reindex(index)
    count_cols = [c for c in feats.columns if c.startswith(("n_", "exit_", "complaint_", "repeat_",
                                                            "competitor_", "last_call_", "has_"))]
    feats[count_cols] = feats[count_cols].fillna(0)
    return feats


class TextRiskScorer:
    """Recency-weighted TF-IDF over customer turns -> logistic churn score per customer."""

    def __init__(self, C=0.5, seed=270):
        self.C, self.seed = C, seed

    def _docs(self, calls, ids):
        texts = calls[calls["customer_text"].str.len() > 0]
        return texts[texts["customer_id"].isin(ids)]

    def _customer_matrix(self, texts, ids):
        X_calls = self.vec.transform(texts["customer_text"])
        w = (texts["recency_weight"] * texts["asr_confidence"]).values
        pos = {c: i for i, c in enumerate(ids)}
        rows = texts["customer_id"].map(pos).values
        agg = sparse.csr_matrix((w, (rows, np.arange(len(rows)))), shape=(len(ids), len(rows)))
        M = agg @ X_calls
        norm = np.asarray(agg.sum(axis=1)).ravel()
        norm[norm == 0] = 1
        return sparse.diags(1 / norm) @ M

    def fit(self, calls, ids, y):
        texts = self._docs(calls, ids)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.6, sublinear_tf=True)
        self.vec.fit(texts["customer_text"])
        ids = [i for i in ids if i in set(texts["customer_id"])]
        X = self._customer_matrix(texts, ids)
        self.model = LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000)
        self.model.fit(X, y.loc[ids])
        return self

    def score(self, calls, ids):
        texts = self._docs(calls, ids)
        have = [i for i in ids if i in set(texts["customer_id"])]
        out = pd.Series(np.nan, index=ids, name="text_risk")
        if have:
            out.loc[have] = self.model.predict_proba(self._customer_matrix(texts, have))[:, 1]
        return out

    def top_terms(self, n=15):
        coef = self.model.coef_[0]
        names = np.array(self.vec.get_feature_names_out())
        order = np.argsort(coef)
        return (pd.DataFrame({"term": names[order[::-1][:n]], "weight": coef[order[::-1][:n]]}),
                pd.DataFrame({"term": names[order[:n]], "weight": coef[order[:n]]}))


def out_of_fold_text_risk(calls, train_ids, y_train, n_splits=5, seed=270):
    """Text score for training customers, each scored by a model that never saw them."""
    callers = [i for i in train_ids if i in set(calls.loc[calls["customer_text"].str.len() > 0, "customer_id"])]
    oof = pd.Series(np.nan, index=train_ids, name="text_risk")
    yc = y_train.loc[callers]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, val_idx in skf.split(callers, yc):
        fit_ids = [callers[i] for i in fit_idx]
        val_ids = [callers[i] for i in val_idx]
        scorer = TextRiskScorer(seed=seed).fit(calls, fit_ids, y_train)
        oof.loc[val_ids] = scorer.score(calls, val_ids).values
    return oof
