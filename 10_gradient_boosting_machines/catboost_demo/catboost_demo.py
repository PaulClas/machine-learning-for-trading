#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import os

from pathlib import Path
import numpy as np
import pandas as pd
import catboost

from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)

with pd.HDFStore('data.h5') as store:
    data = store['data']


def train_test_split(df):
    idx = pd.IndexSlice
    base_cols = ['year', 'month', 'size', 'age']
    df = pd.get_dummies(df,
                        columns=base_cols + ['sector'],
                        prefix=base_cols + [''],
                        prefix_sep=['_'] * 4 + [''])
    train = df.loc[idx[:, '2001-01': '2017-09'], :]
    test = df.loc[idx[:, '2017-10':'2018-03'], :]
    y_train, X_train = train.returns, train.drop('returns', axis=1)
    y_test, X_test = test.returns, test.drop('returns', axis=1)
    return y_train, X_train, y_test, X_test


y_train, X_train, y_test, X_test = train_test_split(data)

label = (data.returns > 1).astype(int)

data = data.drop('returns', axis=1)

print(data.info())

dtrain = xgb.DMatrix(data, label=label)

classifier = xgb.XGBClassifier(max_depth=3,
                               learning_rate=0.1,
                               n_estimators=100,
                               silent=True,
                               objective='binary:logistic',
                               booster='gbtree',
                               n_jobs=-1,
                               gamma=0,
                               min_child_weight=1,
                               max_delta_step=0,
                               subsample=1,
                               colsample_bytree=1,
                               colsample_bylevel=1,
                               reg_alpha=0,
                               reg_lambda=1,
                               scale_pos_weight=1,
                               base_score=0.5,
                               random_state=0)

param_grid = {
    'max_depth'       : [3, 5, 7],
    'learning_rate'   : [.01, .1],
    'n_estimators'    : [50, 200, 500],
    'min_child_weight': [1, 5, 25],
    'gamma'           : [0, 1, 5],
    'subsample'       : [0.8, 1.0],
    'colsample_bytree': [0.7, 1.0],
}


class OneStepTimeSeriesSplit:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the index contains a level labeled 'date'"""

    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        dates = np.sort(X.index.get_level_values('date').unique())[::-1]
        X['idx'] = range(len(X))
        for date in dates[:self.n_splits]:
            train_idx = X.loc[X.index.get_level_values('date') < date, 'idx'].values
            test_idx = X.loc[X.index.get_level_values('date') == date, 'idx'].values
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


cv = OneStepTimeSeriesSplit(n_splits=10)

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=param_grid,
                           cv=cv,
                           scoring='auc',
                           n_jobs=-1,
                           verbose=2)

fit_params = {"early_stopping_rounds": 25,
              "eval_metric"          : 'auc',
              "eval_set"             : [[X_test, y_test]]}

grid_search.fit(X=X_train,
                y=y_train,
                fit_params=fit_params)

joblib.dump(grid_search, 'gridsearch.joblib')
# gs = joblib.load('gridsearch.joblib')

'roc_auc': 'roc_auc',
