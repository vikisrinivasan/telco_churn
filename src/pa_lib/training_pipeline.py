import gc
import pickle

from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

from pa_lib.metric_plot import plot_feature_importance
from pa_lib.params_class import Parameters
import pandas as pd
from typing import *
import os
from xgboost import XGBClassifier

from pa_lib.prediction_pipeline import validation_scores_pipeline
from pa_lib.preprocess_util import null_cleaning, coverage_test, align_and_clean_features, drop_high_correlated_col

FINAL_LOGGING_COLUMNS = ['Model Name', 'Accuracy', 'AUC', 'Lift', 'Precision', 'Recall', 'F1']


def model_training_pipeline(df: pd.DataFrame, params: Parameters, scoring_dict: Dict[str, object], predict_col: str,
                            parameters: str = 'grid_search',
                            save_params: bool = True, sorting_choice: str = 'AUC',
                            grid_score=make_scorer(roc_auc_score), logging: bool = True,
                            logging_path: Optional[str] = None):
    if logging and not logging_path:
        return ValueError('logging set to True but no path provided')
    if logging and not os.path.exists(logging_path):
        return ValueError('logging path does not exists')

    align_features = 0
    nbins = 20
    df[predict_col] = df[predict_col].fillna(0)

    y = df[predict_col]
    if df[predict_col].nunique() == 1:
        return ValueError('the target col %s provided has no variance' % predict_col)

    if 'train_test_split_size' in params.attributes and params.get_param('train_test_split_size'):
        test_split = params.get_param('train_test_split_size')
    else:
        test_split = 0.2

    if ((not isinstance(test_split, float)) and (
            not isinstance(test_split, int)) or test_split >= 1.0 or test_split < 0):
        return ValueError('Please provide the correct value of train test split size')

    df = df.drop(predict_col, axis=1)

    if 'prediction_threshold' in params.attributes and params.get_param('prediction_threshold'):
        try:
            prediction_threshold = float(params.get_param('prediction_threshold'))
            if (prediction_threshold > 1) or (prediction_threshold < 0):
                prediction_threshold = 0.5
        except:
            prediction_threshold = 0.5
    else:
        prediction_threshold = 0.5

    if parameters == 'grid_search':
        if 'CROSSVAL_N' in params.attributes and params.get_param('CROSSVAL_N'):
            try:
                crossval_n = int(params.get_param('CROSSVAL_N'))
                if crossval_n <= 1:
                    crossval_n = 3
            except:
                crossval_n = 3
        else:
            crossval_n = 3

        if 'N_JOBS' in params.attributes and params.get_param('N_JOBS'):
            try:
                jobs_n = int(params.get_param('N_JOBS'))
                if jobs_n < -1 or jobs_n > 10:
                    jobs_n = 1
            except:
                jobs_n = 1
        else:
            jobs_n = 1

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

    if max_threshold is not None or min_threshold is not None:
        low_rm_col, high_rm_col = coverage_test(df, mini=min_threshold, maxi=max_threshold)
        low_rm_col = [a for a in low_rm_col if (a in df.columns)]
        df = df.drop(low_rm_col, axis=1)

        rm_para_col = [a for a in high_rm_col if (a in df.columns)]
        df = df.drop(rm_para_col, axis=1)

        del [max_threshold, min_threshold, low_rm_col, high_rm_col, rm_para_col]

    if 'null_percent' in params.attributes and params.get_param('null_percent'):
        try:
            null_percent = float(params.get_param('null_percent'))
            if null_percent <= 0 or null_percent >= 1.0:
                print("skip the drop null columns")
            else:
                df, null_list = null_cleaning(df, pct_null=null_percent)
                del null_list
                gc.collect()
        except:
            print('skip the drop null columns')

    duplicated_col = df.columns[df.columns.duplicated()]
    if not duplicated_col.empty:
        raise ValueError('Duplicated cols present')
    del [duplicated_col]
    gc.collect()

    if ('corr_threshold' in params.attributes and params.get_param('corr_threshold') is not None):
        correlation_threshold = params.get_param('corr_threshold')
        df = drop_high_correlated_col(df, threshold=correlation_threshold)

    if 'SEED' in params.attributes and params.get_param('SEED') is not None:
        random_state = params.get_param('SEED')
    else:
        random_state = 200

    if 'K_best_features' in params.attributes and params.get_param('K_best_features') is not None:
        try:
            K_FEAT = int(params.get_param('K_best_features'))
        except:
            raise ValueError('Please provide the correct values for K best features')

        if K_FEAT < df.shape[1]:
            selector = SelectKBest(f_classif, k=K_FEAT)
            selector.fit(df, y)
            mask = selector.get_support()
            keep_features = df.columns[mask]
            df = df[keep_features]
            del [keep_features, mask, selector]
            gc.collect()
        else:
            print('Provided too large k best features')

    final_col_list = list(df.columns)
    if logging:
        feature_path = logging_path + '/final_model_feature_list.pkl'
        pkl_out = open(feature_path, 'wb')
        pickle.dump(final_col_list, pkl_out)
        pkl_out.close()
        del [feature_path]
        gc.collect()

    if test_split > 0:
        xtrain, xtest, ytrain, ytest = train_test_split(df, y, stratify=y, test_size=test_split,
                                                        random_state=random_state)

    del df
    del y
    gc.collect()

    if 'sampling' in params.attributes and params.get_param('sampling') == True:
        rus = RandomUnderSampler(random_state=random_state)
        xtrain, ytrain = rus.fit_sample(xtrain, ytrain)
        if test_split > 0 and xtest is not None and ytest is not None:
            xtrain = pd.DataFrame(xtrain, columns=xtest.columns)
            ytrain = pd.Series(ytrain, dtype=ytest.dtype)

    df_fin = pd.DataFrame(data=None, columns=FINAL_LOGGING_COLUMNS)
    if logging:
        if not os.path.exists(logging_path + '/models/'):
            os.makedirs(logging_path + '/models/')
        for key in scoring_dict.keys():
            if not os.path.exists(logging_path + '/models/' + key + "/"):
                os.makedirs(logging_path + '/models/' + key + "/")

    if (align_features == 1):
        xtest = align_and_clean_features(xtest, train_features_list=final_col_list, fill=0)

    for key, est in scoring_dict.items():
        t1 = est

        if parameters == 'grid_search':
            try:
                param_grid = params.get_param('PARAMETER_GRID').get(key)
            except:
                raise ValueError('provide correct values')
            try:

                t1 = GridSearchCV(estimator=est,
                                  cv=crossval_n,
                                  n_jobs=jobs_n,
                                  param_grid=param_grid,
                                  scoring=grid_score,
                                  verbose=10)
                t1.fit(xtrain, ytrain)
                if save_params:
                    if not os.path.exists(logging_path + '/picked_parameters/'):
                        os.makedirs(logging_path + '/picked_parameters/')
                    params_path = logging_path + '/picked_parameters/' + key + '_parameters.pkl'
                    pkl_out = open(params_path, 'wb')
                    pickle.dump(t1.best_params_, pkl_out)
                    pkl_out.close()
            except:
                t1 = est
                t1.fit(xtrain, ytrain)
        else:
            t1.fit(xtrain, ytrain)

        if logging:
            pkl_out = open(logging_path + '/models/' + key + '.pkl', 'wb')
            pickle.dump(t1, pkl_out)
            print("model is saved")
            pkl_out.close()

        if isinstance(est, XGBClassifier) & (isinstance(xtest, pd.DataFrame)):
            try:
                t1_prob_y2 = t1.predict_proba(xtest)
                t1.pred = t1.predict(xtest)
            except:
                t1_prob_y2 = t1.predict_proba(xtest.as_matrix())
                t1.pred = t1.predict(xtest.as_matrix())
        else:
            t1_prob_y2 = t1.predict_proba(xtest)
            t1.pred = t1.predict(xtest)

        if logging:
            new_logging_path = logging_path + "/models/" + key + "/"
        else:
            new_logging_path = logging_path

        validation = pd.DataFrame({'actuals': ytest, 'pred_prob': t1_prob_y2[:, 1]}, columns=['actuals', 'pred_prob'])
        df_t1 = validation_scores_pipeline(validation, logging=logging, path=new_logging_path, TRUE_COL='actuals',
                                           PROB_COL='pred_prob', n_bins=nbins, threshold=prediction_threshold,
                                           model_name=key, FINAL_LOGGING_COLUMNS=FINAL_LOGGING_COLUMNS)

        test_prediction_path = '/test_data_prediction_' + str(key) + '.parquet'
        test_df=xtest.copy()
        test_df['pred_proba']=t1_prob_y2[:,1]
        test_df.to_parquet(logging_path + test_prediction_path)
        del test_prediction_path
        gc.collect()

        plot_feature_importance(t1, final_col_list, logging=logging, path=new_logging_path)

        df_fin = df_fin.append(df_t1)

    df_fin=df_fin.sort_values(by=sorting_choice,ascending=False).reset_index()

    return df_fin
