
import argparse
from pa_lib.training_pipeline import *
from utils.column_names import *
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
def main(args):

    df=pd.read_pickle(os.path.join(args.processed_data_path,'preprocessed_train_df.pkl'))

    params=Parameters(os.path.join(args.parameter_path,'training_parameter.yaml'))


    t5=XGBClassifier(use_label_encoder=False)
    t4=RandomForestClassifier()
    scoring_dict={"XGB":t5,"RF":t4}
    output_path=os.path.join(args.model_logging,'churn_log')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results=model_training_pipeline(df,params,scoring_dict,COL_LABEL,save_params=True,logging=True,logging_path=output_path)
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--processed_data_path', type=str,help='Path for raw data')
    parser.add_argument('--model_logging', type=str, help='Output path for model data')
    parser.add_argument('--parameter_path', type=str, help='Output path for model data')
    parser.add_argument('--teta', type=int, help='Output path for model data')
    args = parser.parse_args()

    main(args)