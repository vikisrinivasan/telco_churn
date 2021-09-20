# -*- coding: utf-8 -*-
from typing import *
from utils.column_names import *
import pandas as pd
from utils.datetime_utils import *
import argparse
import os
from utils.model_config_utils import *

def calculate_total_customer_staying_period(df:pd.DataFrame)-> None:
    df[COL_PROFILE_MONTH]=df[COL_REFERENCE_DATE].apply(lambda x : last_day_of_month(x))
    df[COL_START_DATE]=df[COL_START_DATE].apply(lambda x :  datetime.datetime.strptime(x,"%Y-%m-%d"))
    df[COL_CUST_SINCE]=df.apply(lambda x: diff_month(x[COL_PROFILE_MONTH],x[COL_START_DATE]),axis=1)

def convert_categorical_to_one_hot(df:pd.DataFrame,columns:List[str]) -> None:
    df=pd.get_dummies(df,columns=columns)
    print(df.columns)


def main(args):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    df=pd.read_pickle(os.path.join(args.interim_data_path,'train_df.pkl'))
    calculate_total_customer_staying_period(df)
    convert_categorical_to_one_hot(df,[COL_GENDER,COL_CONTRACT_TYPE,COL_SERVICE_TYPE])
    df.drop(drop_cols,axis=1,inplace=True)
    df.to_pickle(os.path.join(args.processed_data_path, 'preprocessed_train_df.pkl'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--interim_data_path', type=str,help='Path for raw data')
    parser.add_argument('--processed_data_path', type=str, help='Output path for interim data')

    args = parser.parse_args()

    main(args)
