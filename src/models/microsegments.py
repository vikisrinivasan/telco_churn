"""Rule-based microsegments over scored customers.

Rules are ordered and first-match, so every customer lands in exactly one segment.
They were chosen from the churn-rate cuts in the notebook (contract, internet type,
tenure, protection add-ons, pre-T0 call signal) and kept to cuts a retention lead can
read off a CRM screen. Churn Reason is used only to describe segments after the fact;
its mix is nearly the same in every segment, so the actions lean on what can be
observed (contract, add-ons, tenure) and are hypotheses for a holdout test.
"""
import numpy as np
import pandas as pd

REASON_GROUP = {
    "Competitor offered higher download speeds": "Competitor", "Competitor offered more data": "Competitor",
    "Competitor made better offer": "Competitor", "Competitor had better devices": "Competitor",
    "Attitude of support person": "Support experience", "Attitude of service provider": "Support experience",
    "Poor expertise of phone support": "Support experience", "Poor expertise of online support": "Support experience",
    "Network reliability": "Product / network", "Product dissatisfaction": "Product / network",
    "Service dissatisfaction": "Product / network", "Limited range of services": "Product / network",
    "Lack of self-service on Website": "Product / network",
    "Lack of affordable download/upload speed": "Price", "Price too high": "Price",
    "Extra data charges": "Price", "Long distance charges": "Price",
    "Moved": "Not saveable", "Deceased": "Not saveable", "Don't know": "Unknown",
}

SEGMENTS = [
    # (name, rule, action)
    ("Called to complain or leave",
     lambda d: (d["exit_intent_calls"] > 0) | (d["complaint_calls"] > 0),
     "Retention desk call-back within 48h; close the open complaint before any offer"),
    ("New fibre, no support add-ons",
     lambda d: (d["Contract"] == "Month-to-month") & (d["Internet Service"] == "Fiber optic")
     & (d["Tenure Months"] <= 12) & (d["protection_addons"] == 0),
     "Month-2 onboarding call; 3 months of tech support and security free"),
    ("New fibre, with add-ons",
     lambda d: (d["Contract"] == "Month-to-month") & (d["Internet Service"] == "Fiber optic")
     & (d["Tenure Months"] <= 12),
     "Move to a 12-month contract with a first-year discount"),
    ("Established fibre, no contract",
     lambda d: (d["Contract"] == "Month-to-month") & (d["Internet Service"] == "Fiber optic"),
     "12-month price lock in exchange for a contract"),
    ("New DSL, no contract",
     lambda d: (d["Contract"] == "Month-to-month") & (d["Internet Service"] == "DSL")
     & (d["Tenure Months"] <= 12),
     "Fibre upgrade offered on a 12-month contract"),
    ("Other month-to-month",
     lambda d: d["Contract"] == "Month-to-month",
     "No call; email contract offer"),
    ("On 1- or 2-year contract",
     lambda d: pd.Series(True, index=d.index),
     "No outreach before contract end"),
]


def assign_segments(frame):
    seg = pd.Series(pd.NA, index=frame.index, dtype="object")
    for name, rule, _ in SEGMENTS:
        seg = seg.where(seg.notna() | ~rule(frame), name)
    return seg


def segment_summary(frame, score_col="score", capacity=0.20):
    """Size, churn, lift, list share, bill at risk and reason mix per segment.

    `frame` needs: segment, churn, Monthly Charges, Churn Reason and the score column.
    """
    d = frame.copy()
    n_list = int(round(capacity * len(d)))
    d["in_list"] = d[score_col].rank(ascending=False, method="first") <= n_list
    d["reason_group"] = d["Churn Reason"].map(REASON_GROUP)
    base_rate = d["churn"].mean()
    actions = {name: action for name, _, action in SEGMENTS}

    rows = []
    for name, _, _ in SEGMENTS:
        g = d[d["segment"] == name]
        if g.empty:
            continue
        churners = g[g["churn"] == 1]
        mix = churners["reason_group"].value_counts(normalize=True)
        listed = g[g["in_list"]]
        rows.append({
            "segment": name,
            "customers": len(g),
            "share_of_base": len(g) / len(d),
            "churn_rate": g["churn"].mean(),
            "lift_vs_base": g["churn"].mean() / base_rate,
            "in_top20_list": len(listed),
            "list_precision": listed["churn"].mean() if len(listed) else np.nan,
            "churners": len(churners),
            "monthly_bill_lost": churners["Monthly Charges"].sum(),
            "avg_monthly_bill": g["Monthly Charges"].mean(),
            "top_reason": mix.index[0] if len(mix) else "-",
            "top_reason_share": mix.iloc[0] if len(mix) else np.nan,
            "competitor_share": mix.get("Competitor", 0.0),
            "support_share": mix.get("Support experience", 0.0),
            "action": actions[name],
        })
    return pd.DataFrame(rows)
