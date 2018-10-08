#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)

from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)

cat_cols = ['year', 'month', 'age', 'msize', 'sector']


def get_data(start='2000', end='2018', holding_period=1, dropna=True):
    idx = pd.IndexSlice
    target = f'target_{holding_period}m'
    with pd.HDFStore('data.h5') as store:
        df = store['data']

    if start is not None and end is not None:
        df = df.loc[idx[:, start: end], :]
    if dropna:
        df = df.dropna()

    y = (df[target] > 0).astype(int)
    X = df.drop([c for c in df.columns if c.startswith('target')], axis=1)
    return y, X


def get_one_hot_data(df, cols=cat_cols[:-1]):
    df = pd.get_dummies(df,
                        columns=cols + ['sector'],
                        prefix=cols + [''],
                        prefix_sep=['_'] * len(cols) + [''])
    return df.rename(columns={c: c.replace('.0', '') for c in df.columns})


def get_holdout_set(target, features, period=6):
    idx = pd.IndexSlice
    label = target.name
    dates = np.sort(y.index.get_level_values('date').unique())
    cv_start, cv_end = dates[0], dates[-period - 2]
    holdout_start, holdout_end = dates[-period - 1], dates[-1]

    df = features.join(target.to_frame())
    train = df.loc[idx[:, cv_start: cv_end], :]
    y_train, X_train = train[label], train.drop(label, axis=1)

    test = df.loc[idx[:, holdout_start: holdout_end], :]
    y_test, X_test = test[label], test.drop(label, axis=1)
    return y_train, X_train, y_test, X_test


class OneStepTimeSeriesSplit:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the index contains a level labeled 'date'"""

    def __init__(self, n_splits=3, test_period_length=1, shuffle=False):
        self.n_splits = n_splits
        self.test_period_length = test_period_length
        self.shuffle = shuffle
        self.test_end = n_splits * test_period_length

    @staticmethod
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def split(self, X, y=None, groups=None):
        unique_dates = (X
                            .index
                            .get_level_values('date')
                            .unique()
                            .sort_values(ascending=False)
        [:self.test_end])

        dates = X.reset_index()[['date']]
        for test_date in self.chunks(unique_dates, self.test_period_length):
            train_idx = dates[dates.date < min(test_date)].index
            test_idx = dates[dates.date.isin(test_date)].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


lgb_clf = LGBMClassifier(boosting_type='gbdt',
                         device='gpu',
                         objective='binary',
                         metric='auc',
                         num_leaves=120,
                         max_depth=5,
                         learning_rate=0.05,
                         n_estimators=200,
                         subsample_for_bin=200000,
                         class_weight=None,
                         min_split_gain=0.0,
                         min_child_weight=0.001,
                         min_child_samples=20,
                         subsample=1.0,
                         subsample_freq=0,
                         colsample_bytree=1.0,
                         reg_alpha=0.0,
                         reg_lambda=0.0,
                         random_state=42,
                         n_jobs=-1,
                         silent=True,
                         importance_type='gain'
                         )
param_grid = {
    'max_depth'       : [3, 5, 8, 12],
    'num_leaves'      : [20, 35, 50, 100],
    'n_estimators'    : [100, 200, 300],
}

n_splits = 10
cv = OneStepTimeSeriesSplit(n_splits=n_splits)

y, features = get_data()
X = get_one_hot_data(features)

grid_search = GridSearchCV(estimator=lgb_clf,
                           param_grid=param_grid,
                           cv=cv,
                           scoring='roc_auc',
                           n_jobs=-1,
                           verbose=1)

grid_search.fit(X=X,
                y=y)

joblib.dump(grid_search, 'lgb_gridsearch.joblib')
# gs = joblib.load('gridsearch.joblib')

# pip
# install
# lightgbm - -install - option = --gpu - -install - option = "--opencl-include-dir=/usr/local/cuda-9.0/include/ " - -install - option = "--opencl-library=/usr/local/cuda-9.0/lib64/libOpenCL.so"
