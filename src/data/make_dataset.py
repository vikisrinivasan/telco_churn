# -*- coding: utf-8 -*-
import logging
import string
from pathlib import Path

from nltk import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from utils.column_names import *
import pandas as pd
import calendar
from utils.feature_utils import tokenize_text
from typing import *
import argparse
import os


def cdr_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    df[COL_YEAR] = pd.DatetimeIndex(df[COL_DATE]).year
    df[COL_MONTH] = pd.DatetimeIndex(df[COL_DATE]).month
    df[COL_MONTH_NAME] = df[COL_MONTH].apply(lambda x: calendar.month_abbr[x])
    cdr_record_cols = [col for col in df.columns if
                       col not in [COL_MSISDN, COL_DATE, COL_YEAR, COL_MONTH, COL_MONTH_NAME]]
    cdr_per_month = df.groupby([COL_MSISDN, COL_YEAR, COL_MONTH_NAME])[cdr_record_cols].sum().reset_index()
    cdr_per_month = cdr_per_month.pivot_table(index=COL_MSISDN, columns=[COL_YEAR, COL_MONTH_NAME],
                                              values=cdr_record_cols).reset_index()

    columns=[COL_MSISDN]+["%s_%s_%s"%(col[0],col[1],col[2]) for col in cdr_per_month.columns if col[1]!='']
    cdr_per_month.columns=columns

    return cdr_per_month


def get_wordnet_pos(tag: str) -> str:
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_text(text: str) -> str:
    text = text.lower()
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    text = [word for word in text if not any(c.isdigit() for c in word)]
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    text = [t for t in text if len(t) > 0]
    pos_tags = pos_tag(text)
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    text = [t for t in text if len(t) > 1]
    text = " ".join(text)
    return text


def aggregate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df[COL_YEAR] = pd.DatetimeIndex(df[COL_DATE]).year
    df[COL_MONTH] = pd.DatetimeIndex(df[COL_DATE]).month
    df[COL_MONTH_NAME] = df[COL_MONTH].apply(lambda x: calendar.month_abbr[x])
    df[COL_FEEDBACK_CLEAN] = df[COL_FEEDBACK].apply(lambda x: clean_text(x))
    sid = SentimentIntensityAnalyzer()
    df[COL_SENTIMENTS] = df[COL_FEEDBACK].apply(lambda x: sid.polarity_scores(x))
    df = pd.concat([df.drop([COL_SENTIMENTS], axis=1), df[COL_SENTIMENTS].apply(pd.Series)], axis=1)
    df[COL_COMP_SCORE] = df[COL_COMPOUND].apply(lambda c: 'pos' if c >= 0 else 'neg')
    sentiment_scores = pd.get_dummies(df[COL_COMP_SCORE])
    final_df = df.drop([COL_NEG, COL_NEU, COL_POS, COL_COMPOUND], axis=1)
    final_df = final_df.join(sentiment_scores)

    final_df_vect = tokenize_text(COL_FEEDBACK_CLEAN, final_df)
    final_df = final_df.join(final_df_vect)
    print(final_df.head(10))
    final_aggregation = final_df.drop([COL_DATE, COL_FEEDBACK, COL_FEEDBACK_CLEAN], axis=1)
    final_df_score = final_aggregation.groupby([COL_MSISDN])[
        list(final_df_vect.columns) + [COL_NEG, COL_POS]].sum().reset_index()
    return final_df_score


def get_competitor_flag(df: pd.DataFrame, competitors: List[str]) -> None:
    df.loc[df[COL_WEBSITE].str.contains("|".join(competitors)), COL_COMPETITOR] = True
    df.fillna(False, inplace=True)


def aggregate_ddmr(df: pd.DataFrame) -> pd.DataFrame:
    df[COL_YEAR] = pd.DatetimeIndex(df[COL_DATE]).year
    df[COL_MONTH] = pd.DatetimeIndex(df[COL_DATE]).month
    df[COL_MONTH_NAME] = df[COL_MONTH].apply(lambda x: calendar.month_abbr[x])
    df[COL_WEBSITE] = df[COL_WEBSITE].str.split(".").apply(lambda x: x[1])
    get_competitor_flag(df, ['singtel', 'starhub'])

    df_total_visits_per_month = df.groupby([COL_MSISDN, COL_YEAR, COL_MONTH_NAME]).agg(
        total_visits=(COL_VISITS, 'sum')).reset_index()
    df_total_visits_per_year_per_website = df.groupby([COL_MSISDN, COL_YEAR, COL_MONTH_NAME, COL_WEBSITE]).agg(
        total_visits=(COL_VISITS, 'sum')).reset_index()
    df_total_visits_per_year_per_competitor = df[df[COL_COMPETITOR] == True].groupby(
        [COL_MSISDN, COL_YEAR, COL_MONTH_NAME, COL_COMPETITOR]).agg(total_visits=(COL_VISITS, 'sum')).reset_index()
    df_total_visits_per_year_per_others = df[df[COL_COMPETITOR] == False].groupby(
        [COL_MSISDN, COL_YEAR, COL_MONTH_NAME, COL_COMPETITOR]).agg(total_visits=(COL_VISITS, 'sum')).reset_index()

    # pivoting

    df_competitor = df_total_visits_per_year_per_competitor.pivot_table(index=COL_MSISDN, columns=COL_MONTH_NAME,
                                                                        values=COL_TOTAL_VISITS).reset_index()
    df_website = df_total_visits_per_year_per_website.pivot_table(index=COL_MSISDN,
                                                                  columns=[COL_MONTH_NAME, COL_WEBSITE],
                                                                  values=COL_TOTAL_VISITS).reset_index()
    df_total = df_total_visits_per_month.pivot_table(index=COL_MSISDN, columns=COL_MONTH_NAME,
                                                     values=COL_TOTAL_VISITS).reset_index()
    df_others = df_total_visits_per_year_per_others.pivot_table(index=COL_MSISDN, columns=COL_MONTH_NAME,
                                                                values=COL_TOTAL_VISITS).reset_index()

    df_website.columns = [COL_MSISDN] + ["%s_%s" % (col[0], col[1]) for col in df_website.columns if col[1] != '']

    df_competitor.columns = [COL_MSISDN] + ["%s_competitor_visits" % (col.lower()) for col in df_competitor.columns if
                                            col != COL_MSISDN]

    df_others.columns = [COL_MSISDN] + ["%s_non_competitor_visits" % (col.lower()) for col in df_others.columns if
                                        col != COL_MSISDN]

    df_total.columns = [COL_MSISDN] + ["%s_visits" % (col.lower()) for col in df_total.columns if col != COL_MSISDN]

    return df_competitor.merge(df_website, on=[COL_MSISDN],
                               how="inner") \
        .merge(df_others, on=[COL_MSISDN], how="inner") \
        .merge(df_total, on=[COL_MSISDN], how="inner")


def main(args):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    cdr_df = pd.read_csv(os.path.join(args.raw_data_path, 'telco_cdr.tsv.gz'), compression='gzip', header=0, sep='\t',
                         quotechar='"')
    census_df = pd.read_csv(os.path.join(args.raw_data_path, 'telco_census.tsv.gz'), compression='gzip', header=0,
                            sep='\t',
                            quotechar='"')

    reviews_df = pd.read_csv(os.path.join(args.raw_data_path, 'telco_reviews.tsv.gz'), compression='gzip', header=0,
                             sep='\t', quotechar='"')

    ddmr_df = pd.read_csv(os.path.join(args.raw_data_path, 'telco_web.tsv.gz'), compression='gzip', header=0, sep='\t',
                          quotechar='"')
    train_df = pd.read_csv(os.path.join(args.raw_data_path, 'telco_train.tsv.gz'), compression='gzip', header=0,
                           sep='\t',
                           quotechar='"')

    cdr_final_df = cdr_aggregation(cdr_df)
    reviews_final_df = aggregate_reviews(reviews_df)
    ddmr_final_df = aggregate_ddmr(ddmr_df)

    train_df = train_df.merge(ddmr_final_df, on=[COL_MSISDN], how='left'). \
        merge(cdr_final_df, on=[COL_MSISDN], how='left'). \
        merge(reviews_final_df, on=[COL_MSISDN], how='left')

    train_df = train_df.merge(census_df, on=[COL_PLANNING_AREA], how='left')

    train_df.to_pickle(os.path.join(args.interim_data_path, 'train_df.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--raw_data_path', type=str, help='Path for raw data')
    parser.add_argument('--interim_data_path', type=str, help='Output path for interim data')

    args = parser.parse_args()

    main(args)
