"""Customer-level tabular features, all observable on the scoring date."""
import numpy as np
import pandas as pd

CATEGORICAL = ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service",
               "Multiple Lines", "Internet Service", "Online Security", "Online Backup",
               "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
               "Contract", "Paperless Billing", "Payment Method"]
NUMERIC = ["Tenure Months", "Monthly Charges", "Total Charges"]

ADDONS = ["Online Security", "Online Backup", "Device Protection", "Tech Support",
          "Streaming TV", "Streaming Movies"]


# Total Charges accumulates up to the last account update, which can fall after T0.
# It adds nothing measurable (notebook section 2.3), so the default feature set drops it.
POST_T0_RISK = ["Total Charges", "avg_monthly_billed", "bill_vs_history"]


def build_tabular(customers, t0_safe=True):
    """Return a feature frame indexed by CustomerID.

    Engineered columns:
      n_addons            - count of add-on services held (stickiness)
      protection_addons   - security + backup + device protection + tech support
      avg_monthly_billed  - Total Charges / tenure: what the customer has paid on average
      bill_vs_history     - current Monthly Charges minus that average; > 0 means the bill rose
      new_customer        - tenure under 6 months
    """
    df = customers.set_index("CustomerID")
    X = df[CATEGORICAL + NUMERIC].copy()
    X["n_addons"] = (df[ADDONS] == "Yes").sum(axis=1)
    X["protection_addons"] = (df[ADDONS[:4]] == "Yes").sum(axis=1)
    tenure = df["Tenure Months"].replace(0, np.nan)
    X["avg_monthly_billed"] = (df["Total Charges"] / tenure).fillna(df["Monthly Charges"])
    X["bill_vs_history"] = df["Monthly Charges"] - X["avg_monthly_billed"]
    X["new_customer"] = (df["Tenure Months"] < 6).astype(int)
    for col in CATEGORICAL:
        X[col] = X[col].astype("category")
    return X.drop(columns=POST_T0_RISK) if t0_safe else X


def tenure_band(months):
    return pd.cut(months, [-1, 6, 12, 24, 48, 100],
                  labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"])
