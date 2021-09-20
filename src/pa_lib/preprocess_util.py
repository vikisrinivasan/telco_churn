import gc
import pickle

import pandas as pd
from typing import *

from numpy import corrcoef


def get_feature_correlated(X:pd.DataFrame,threshold:float):
    X=X.select_dtypes(exclude='object')

    if X.isna().values.any():
        X=X.fillna(0)

    corr=corrcoef(X,rowvar=False)
    corr=pd.DataFrame(corr).unstack()
    pairs_to_drop=set()
    for i in range(0,X.shape[1]):
        for j in range(0,i+1):
            pairs_to_drop.add((i,j))

    corr=corr.drop(labels=pairs_to_drop)
    corr=corr.reindex(corr.abs().sort_values(ascending=False).index)
    corr=corr[corr.abs().sort_values(ascending=False).index]
    corr=corr[corr.abs()>threshold].reset_index()
    if corr.shape[0]>0:
        corr.iloc[:,:2]=X.columns[corr.iloc[:,:2]]

    return corr

def drop_high_correlated_col(df:pd.DataFrame,threshold:float=0.8) ->pd.DataFrame:
    corr_cols=get_feature_correlated(df,threshold)
    remove_col=[a for a in corr_cols.level_1.unique() if a in df.columns]
    df=df.drop(columns=remove_col,axis=1)
    return df





def coverage_test(df: pd.DataFrame, mini: float = None, maxi: float = None) -> Tuple[List[str], List[str]]:
    unique_count = df.apply(lambda x: x.nunique(), axis=0)
    object_columns = list(df.select_dtypes(include='object').columns)
    if mini:
        no_info_col = unique_count[unique_count < mini].index.values.tolist()
    else:
        no_info_col = []
    if maxi:
        high_nums = [a for a in unique_count[unique_count > maxi].index.values.tolist() if a in object_columns]
    else:
        high_nums = []
    return no_info_col, high_nums


def null_cleaning(df: pd.DataFrame, pct_null: float = 0.8) -> Tuple[pd.DataFrame, List[str]]:
    null_series = df.isnull().sum().sort_values(ascending=False)
    null_list = list(null_series[null_series > pct_null * len(df)].index)
    df.drop(null_list, axis=1, inplace=True)
    return df, null_list


def align_and_clean_features(features: pd.DataFrame, feature_names_path: Optional[str] = None,
                             train_features_list: Optional[List[str]] = None, fill: int = 0) -> pd.DataFrame:
    curr_features = features.columns.to_list()
    duplicated_col = features.columns[features.columns.duplicated()]

    if not duplicated_col.empty:
        raise ValueError('Duplicated cols')
    del [duplicated_col]
    gc.collect()

    if train_features_list is None:
        try:
            with open(feature_names_path, 'rb') as f:
                train_features = pickle.load(f)
        except:
            raise ValueError("Please provide a valid path")
    else:
        train_features = train_features_list

    if not isinstance(train_features, list):
        raise NotImplementedError("Only list of training feature names supported")

    if curr_features != train_features:
        for true_col in train_features:
            if true_col not in curr_features:
                features[true_col] = fill
        extra_cols = [col for col in curr_features if col not in train_features]
        features = (features.drop(columns=extra_cols).reindex(train_features, axis='columns'))
        del [extra_cols]
        gc.collect()

    del [curr_features, train_features]
    gc.collect()
    return features
