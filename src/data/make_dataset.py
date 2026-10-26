# -*- coding: utf-8 -*-
"""Step 2 of the pipeline: clean the raw sources and join calls to customers.

Inputs (data/raw):  Telco_customer_churn.xlsx, Telco_Customer_Interactions.xlsx
Outputs (data/interim):
    customers.pkl        one row per customer, Total Charges fixed, `churn` label added
    calls.pkl            calls observable at T0, joined to a known customer, de-duplicated
    calls_set_aside.pkl  every dropped call with the reason it was dropped
    data_audit.json      row counts for each cleaning step

Every cleaning step is logged so the notebook can report what was changed and why.
"""
import argparse
import json
import os

import pandas as pd

from src.data.extract_dataset import CALLS_FILE, CUSTOMERS_FILE

T0 = pd.Timestamp("2026-03-31")

# Columns produced by IBM after (or because of) the outcome. They must never be model inputs.
OUTCOME_COLUMNS = ["Churn Label", "Churn Value", "Churn Score", "CLTV", "Churn Reason"]
# Constant or identifier columns with no transferable signal.
NON_FEATURE_COLUMNS = ["CustomerID", "Count", "Country", "State", "City", "Zip Code",
                       "Lat Long", "Latitude", "Longitude"]

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")


def load_customers(raw_data_path: str = RAW_DIR):
    """Return the cleaned customer table and an audit list of (issue, rows, action)."""
    df = pd.read_excel(os.path.join(raw_data_path, CUSTOMERS_FILE))
    audit = []

    # Total Charges is text; 11 brand-new customers (tenure 0) have a blank value.
    blank = df["Total Charges"].astype(str).str.strip() == ""
    audit.append(("Blank Total Charges (all tenure 0, none churned)", int(blank.sum()),
                  "Set to 0: no bill has been issued yet"))
    df["Total Charges"] = pd.to_numeric(df["Total Charges"].where(~blank, 0))

    for col in ["Count", "Country", "State"]:
        audit.append((f"{col} has one value ({df[col].iloc[0]})", len(df), "Dropped from features"))
    audit.append(("Churn Score / CLTV are IBM model outputs built with the outcome in view",
                  len(df), "Excluded from features; CLTV kept only as a value lens"))
    audit.append(("Churn Reason is recorded at exit (churners only)", int(df["Churn Reason"].notna().sum()),
                  "Excluded from features; used only to read the ranking"))

    df["Churn Reason"] = df["Churn Reason"].fillna("")
    df["churn"] = df["Churn Value"].astype(int)
    return df, audit


def clean_customer_id(series: pd.Series) -> pd.Series:
    """Normalise the free-typed join key: strip spaces and upper-case."""
    return series.astype(str).str.strip().str.upper()


def load_raw_calls(raw_data_path: str = RAW_DIR) -> pd.DataFrame:
    return pd.read_excel(os.path.join(raw_data_path, CALLS_FILE), sheet_name="interactions")


def load_calls(customer_ids, raw_data_path: str = RAW_DIR, t0: pd.Timestamp = T0):
    """Join calls to customers and keep what a retention team could see at T0.

    Returns (usable_calls, set_aside, audit).
    """
    calls = load_raw_calls(raw_data_path)
    calls["customer_id_raw"] = calls["customer_id"]
    calls["customer_id"] = clean_customer_id(calls["customer_id"])
    audit, set_aside = [], []

    def drop(mask, reason):
        nonlocal calls
        if mask.sum():
            set_aside.append(calls[mask].assign(set_aside_reason=reason))
            audit.append((reason, int(mask.sum())))
        calls = calls[~mask].copy()

    trimmed = calls["customer_id_raw"].astype(str) != calls["customer_id"]
    audit.append(("customer_id had leading/trailing spaces (fixed, kept)", int(trimmed.sum())))

    # Inner join on the cleaned key: calls for customers not in the customer file are orphans.
    drop(~calls["customer_id"].isin(customer_ids), "Customer ID not in customer file (orphan)")
    drop(calls["interaction_date"] > t0, "After scoring date T0 2026-03-31 (not observable)")

    # The extract is not de-duplicated: the same call logged twice for one customer.
    dup = calls.duplicated(["customer_id", "transcript"], keep="first") & calls["transcript"].notna()
    drop(dup, "Duplicate log of the same call")

    # Empty or stub transcripts carry no words; keep them as a failed-contact count instead.
    stub = calls["transcript"].isna() | (calls["transcript"].str.len() < 60)
    calls["is_stub"] = stub
    audit.append(("Empty or stub transcript ([call disconnected], [no audio captured]); "
                  "text ignored, counted as a failed contact", int(stub.sum())))

    # interaction_seq was computed on the raw, duplicated extract; rebuild it.
    calls = calls.sort_values(["customer_id", "interaction_date", "interaction_id"])
    calls["interaction_seq"] = calls.groupby("customer_id").cumcount() + 1

    set_aside = pd.concat(set_aside, ignore_index=True) if set_aside else pd.DataFrame()
    return calls.reset_index(drop=True), set_aside, audit


def main(args):
    os.makedirs(args.interim_data_path, exist_ok=True)
    customers, customer_audit = load_customers(args.raw_data_path)
    calls, set_aside, call_audit = load_calls(set(customers["CustomerID"]), args.raw_data_path)

    customers.to_pickle(os.path.join(args.interim_data_path, "customers.pkl"))
    calls.to_pickle(os.path.join(args.interim_data_path, "calls.pkl"))
    set_aside.to_pickle(os.path.join(args.interim_data_path, "calls_set_aside.pkl"))
    with open(os.path.join(args.interim_data_path, "data_audit.json"), "w") as f:
        json.dump({"customers": customer_audit, "calls": call_audit}, f, indent=1)

    print(f"customers {len(customers):,} | usable calls {len(calls):,} from "
          f"{calls['customer_id'].nunique():,} customers | set aside {len(set_aside):,}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean raw data and join calls to customers.')
    parser.add_argument('--raw_data_path', type=str, default=RAW_DIR, help='Path for raw data')
    parser.add_argument('--interim_data_path', type=str, default=INTERIM_DIR, help='Output path for interim data')
    main(parser.parse_args())
