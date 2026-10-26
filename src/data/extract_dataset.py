# -*- coding: utf-8 -*-
"""Step 1 of the pipeline: download the two raw sources into data/raw.

    IBM Telco Customer Churn (Kaggle)       -> Telco_customer_churn.xlsx
    Contact-centre transcripts (HuggingFace) -> Telco_Customer_Interactions.xlsx

Files already present are left untouched, so the raw layer stays immutable.
"""
import argparse
import io
import os
import urllib.request
import zipfile

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/yeanzc/telco-customer-churn-ibm-dataset"
HF_URL = ("https://huggingface.co/datasets/mepriyak85/Public_Call_Transcripts_Data_Kaggle"
          "/resolve/main/Telco_Customer_Interactions.xlsx")

CUSTOMERS_FILE = "Telco_customer_churn.xlsx"
CALLS_FILE = "Telco_Customer_Interactions.xlsx"


def extract_customers(raw_data_path: str) -> str:
    path = os.path.join(raw_data_path, CUSTOMERS_FILE)
    if not os.path.exists(path):
        with urllib.request.urlopen(KAGGLE_URL) as resp:
            zipfile.ZipFile(io.BytesIO(resp.read())).extract(CUSTOMERS_FILE, raw_data_path)
    return path


def extract_calls(raw_data_path: str) -> str:
    path = os.path.join(raw_data_path, CALLS_FILE)
    if not os.path.exists(path):
        urllib.request.urlretrieve(HF_URL, path)
    return path


def main(args):
    os.makedirs(args.raw_data_path, exist_ok=True)
    for path in (extract_customers(args.raw_data_path), extract_calls(args.raw_data_path)):
        print(f"{path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download raw Singtel case-study data.')
    parser.add_argument('--raw_data_path', type=str, help='Output path for raw data')
    main(parser.parse_args())
