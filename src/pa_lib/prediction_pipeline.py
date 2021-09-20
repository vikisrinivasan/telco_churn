import gc
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from pa_lib.metric_plot import roc_plot, plot_lift_chart, plot_coverage_chart
from pa_lib.params_class import Parameters
import pandas as pd
from typing import *

from pa_lib.preprocess_util import coverage_test, align_and_clean_features

FINAL_LOGGING_COLUMNS = ['Model Name', 'Accuracy', 'AUC', 'Lift', 'Precision', 'Recall', 'F1']


def prediction_pipeline(df: pd.DataFrame, model_path: str, feature_list_path: str, params: Parameters) -> np.ndarray:
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print('error loading model')
        raise e

    if 'columns_to_drop' in params.attributes and params.get_param('columns_to_drop'):
        rm_para_col = params.get_param('columns_to_drop')
    else:
        rm_para_col = []

    if len(rm_para_col) > 0:
        for col in rm_para_col:
            if (col in df.columns):
                df = df.drop(col, axis=1)
    max_threshold = None
    min_threshold = None

    if 'density_threshold_max' in params.attributes and params.get_param('density_threshold_max'):
        try:

            max_threshold = params.get_param('density_threshold_max')
        except:
            raise ValueError('please provide correct value')

    if 'density_threshold_min' in params.attributes and params.get_param('density_threshold_min'):
        try:

            min_threshold = params.get_param('density_threshold_min')
        except:
            raise ValueError('please provide correct value')

    low_rm_col, high_rm_col = coverage_test(df, mini=min_threshold, maxi=max_threshold)
    low_rm_col = [a for a in low_rm_col if (a in df.columns)]
    df = df.drop(low_rm_col, axis=1)

    rm_para_col = [a for a in high_rm_col if (a in df.columns)]
    df = df.drop(rm_para_col, axis=1)

    del [max_threshold, min_threshold, low_rm_col, high_rm_col, rm_para_col]

    duplicated_col = df.columns[df.columns.duplicated()]
    if not duplicated_col.empty:
        raise ValueError('Duplicated cols present')
    del [duplicated_col]
    gc.collect()

    features = align_and_clean_features(df, feature_names_path=feature_list_path, fill=0)
    proba = model.predict_proba(features)[:, 1]
    if model.classes_.size > 2:
        proba = model.predict_proba(features)[:, 1:]
    return proba


def proba_confusion_matrix(df, actual_col, proba_col, threshold):
    if df[[actual_col, proba_col]].isnull().values.any():
        df[actual_col] = df[actual_col].fillna(0)
        df[proba_col] = df[proba_col].fillna(0)
    binary_col = df[proba_col] > threshold
    if (binary_col.sum() > 0) & (binary_col.sum() < len(binary_col)):
        tn, fp, fn, tp = confusion_matrix(df[actual_col], binary_col).ravel()
        matrix = pd.DataFrame({'Predicted No': [tn, fn], 'Predicted Yes': [tp, fp]},
                              index=['Actual No:', 'Actual Yes:'])
        precision = precision_score(df[actual_col], binary_col)
        recall = recall_score(df[actual_col], binary_col)
        f1 = f1_score(df[actual_col], binary_col)
        accuracy = accuracy_score(df[actual_col], binary_col)
    else:
        matrix = pd.DataFrame({'Predicted No': [-1, -1], 'Predicted Yes': [-1, -1]},
                              index=['Actual No:', 'Actual Yes:'])
        precision = -1
        recall = -1
        f1 = -1
        accuracy = -1
    del [binary_col]
    gc.collect()
    return matrix, precision, recall, f1, accuracy


def validation_scores_pipeline(df: pd.DataFrame, logging: bool = False, path: Optional[str] = None,
                               TRUE_COL: str = 'actuals', PROB_COL: str = 'probabilities',
                               n_bins: int = 10, threshold: float = 0.5, model_name: str = 'XGB',
                               FINAL_LOGGING_COLUMNS: List[str]
                               = ['Model Name', 'Accuracy', 'AUC', 'Lift', 'Precision', 'Recall', 'F1']) -> pd.DataFrame:
    if df[[TRUE_COL, PROB_COL]].isnull().values.any():
        df[TRUE_COL] = df[TRUE_COL].fillna(0)
        df[PROB_COL] = df[PROB_COL].fillna(0)

    df = df.reset_index(drop=True)

    auc = roc_plot(df, path=path, TRUE_COL=TRUE_COL, PROB_COL=PROB_COL, logging=logging)

    plot_coverage_chart(df,prob_col=PROB_COL,logging=logging,path=path)

    lift = plot_lift_chart(df, logging=logging, TRUE_COL=TRUE_COL, PROB_COL=PROB_COL, path=path, n_bins=n_bins)

    thresholds = [threshold,
                  np.percentile(df[PROB_COL], 90),
                  np.percentile(df[PROB_COL], 80)]
    index = ['Overall', 'Bin 1', 'Bin 2']

    df_t1 = pd.DataFrame(columns=FINAL_LOGGING_COLUMNS)
    accuracy, precision, recall, f1 = (0, 0, 0, 0)
    for i, thresh in enumerate(thresholds):
        matrix, precision, recall, f1, accuracy = \
            proba_confusion_matrix(df, TRUE_COL, PROB_COL, threshold=thresh)
        lift_bin = lift[0]
        if i == 2:
            lift_bin = (lift[0] + lift[1]) / 2
        df_t1 = df_t1.append(
            pd.DataFrame([[model_name, accuracy, auc, lift_bin, precision, recall, f1]], columns=FINAL_LOGGING_COLUMNS))

    df_t1.index = index
    del [auc, lift, accuracy, precision, recall, f1]
    gc.collect()
    return df_t1.drop(index=['Bin 1', 'Bin 2'])
