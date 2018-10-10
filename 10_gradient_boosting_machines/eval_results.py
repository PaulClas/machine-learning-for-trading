#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)

cols = ['model', 'data', 'train', 'valid', 'boosting', 'learning_rate', 'max_depth', 'num_leaves',
        'colsample_bytree', 'min_gain_to_split', 'rounds', 'time']

with pd.HDFStore('results.h5') as store:
    xgb = store['xgboost/dummies'].rename(columns={'booster': 'boosting', 'gamma': 'min_gain_to_split'}).assign(
            model='xgb', data='dummies')
    print(xgb.info())
    lgb_dummies = store['lightgbm/dummies'].assign(model='lightgbm', data='dummies')
    lgb_dummies.boosting = lgb_dummies.boosting.str.replace('gbdt', 'gbtree')
    print(lgb_dummies.info())
    lgb_factors = store['lightgbm/factors'].assign(model='lightgbm', data='factors')
    print(lgb_factors.info())

    df = (lgb_dummies
          .append(lgb_factors)
          .append(xgb, ignore_index=True)).loc[:, cols]
    print(df.info())
    store.put('xgb_lgb', df)
    # valid = df.filter(like='valid')
    # print(valid.corr('spearman'))
    # print(valid.describe())


def results_overview():
    with pd.HDFStore('model_tuning.h5') as store:
        df = store['xgboost/dummies/results']
        print(df.sort_values('valid', ascending=False).head())


def get_params():
    path = Path('params')
    with pd.ExcelWriter(path / 'params.xlsx') as xls:
        for model in ['lightgbm', 'xgboost', 'catboost']:
            df = pd.read_csv(path / (model + '.txt'), header=None, names=['parameter', 'default'])
            print(df)
            df.to_excel(xls, sheet_name=model, index=False)


def clean_results():
    with pd.HDFStore('model_tuning.h5') as store:
        print(store.info())
        results = [k for k in store.keys() if k.endswith('results')]
        for result in results:
            print('\n', result)
            df = store[result]
            # drop info that doesn't change across models
            df = df.loc[:, ~df.eq(df.iloc[0, :], axis=1).all()].dropna(how='all', axis=1)
            print(df.info())
            with pd.HDFStore('results.h5') as target:
                target.put('/'.join(result[1:].split('/')[:2]),
                           df.sort_values('valid', ascending=False).reset_index(drop=True))


clean_results()
exit()


def summarize_result():
    df = df[~df.eq(df.iloc[:, 0], axis=0).all(1)]
    params = df.loc[['params', 'time'], :]
    df = df.drop(['params', 'time'], level=0).apply(pd.to_numeric)
    params = params.append(df.groupby(level=0).mean())
    print(params.T.sort_values('valid', ascending=False))


# with pd.HDFStore('results.h5') as store:
# print(store.info())
# exit()
# for k in store.keys():
#     print('\n', k)
#     df = store[k]
#     print(df.sort_values('valid', ascending=False).head(10))

with pd.HDFStore('results.h5') as store:
    df = store['xgboost/dummies']
    for f in ['learning_rate', 'min_split_loss', 'max_depth', 'colsample_bytree', 'boosting']:
        print(df.groupby(f).valid.mean().sort_values(ascending=False))

