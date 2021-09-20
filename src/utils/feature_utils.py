from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd



def tokenize_text(col_name: str, df: pd.DataFrame) -> pd.DataFrame:
    count_vect = CountVectorizer()
    feedback_vect = count_vect.fit_transform(df[col_name])
    feedback_vect_arr = feedback_vect.toarray()
    review_vect = pd.DataFrame(data=feedback_vect_arr,
                               columns=["%s_%s" % (feat_nm, 'word') for feat_nm in count_vect.get_feature_names()])
    return review_vect
