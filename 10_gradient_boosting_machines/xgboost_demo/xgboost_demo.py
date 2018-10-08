#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.callback import reset_learning_rate
from time import time

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)

cat_cols = ['year', 'month', 'age', 'msize', 'sector']


def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>2.1f}'


def learning_rate(n, ntot):
    start_eta = 0.1
    k = 8 / ntot
    x0 = ntot / 1.8
    return start_eta * (1 - 1 / (1 + np.exp(-k * (n - x0))))


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


n_splits = 10
cv = OneStepTimeSeriesSplit(n_splits=n_splits)

y, features = get_data()
X = get_one_hot_data(features)

y_train, X_train, y_test, X_test = get_holdout_set(target=y, features=X, period=6)

with pd.HDFStore('model_tuning.h5') as store:
    store.put('xgboost/holdout/features', X_test)
    store.put('xgboost/holdout/target', y_test)

datasets = {}
for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
    datasets[fold] = {'train': xgb.DMatrix(label=y.iloc[train_idx], data=X.iloc[train_idx], nthread=-1),
                      'eval' : xgb.DMatrix(label=y.iloc[test_idx], data=X.iloc[test_idx], nthread=-1)}

params = dict(
        objective='binary:logistic',
        eval_metric=['logloss', 'auc'],
        # tree_method='gpu_hist',
        n_jobs=-1,
        silent=1,
        seed=42
)
boost_params = dict(
        eta=0.3,
        gamma=0,
        max_depth=6,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        alpha=0
)
boost_params['lambda'] = 1  # reserved keyword
params.update(boost_params)

cv_sets = ['train', 'eval']
boost_rounds = 250
early_stopping = 25

results = ['learning_curve', 'feature_importance', 'cv_result']
cv_results = {result: pd.DataFrame() for result in results}
models = pd.DataFrame()


def run_cv(test_params, data, n_splits=10):
    df = {r: pd.DataFrame() for r in results}

    for fold in range(n_splits):
        dtrain = data[fold]['train']
        dtest = data[fold]['eval']

        watchlist = list(zip([dtrain, dtest], cv_sets))
        scores = {}

        model = xgb.train(params=test_params,
                          dtrain=dtrain,
                          num_boost_round=boost_rounds,
                          verbose_eval=50,
                          evals=watchlist,
                          early_stopping_rounds=early_stopping,
                          evals_result=scores,
                          callbacks=[reset_learning_rate(learning_rate)])

        df['cv_result'] = pd.concat([df['cv_result'],
                                     pd.DataFrame({
                                         'predictions': model.predict(dtest),
                                         'actuals'    : dtest.get_label(),
                                     }).assign(fold=fold)])

        df['learning_curve'] = pd.concat([df['learning_curve'],
                                          (pd.concat([pd.DataFrame(scores[s]) for s in cv_sets],
                                                     axis=1,
                                                     keys=cv_sets)
                                           .assign(fold=fold))])

        df['feature_importance'] = (pd.concat([df['feature_importance'],
                                               pd.Series(model.get_score(importance_type='gain'))
                                              .to_frame('fi')
                                              .assign(fold=fold)]))
    return df


boosters = ['gbtree', 'dart']
# tree_methods = ['gpu_exact', 'gpu_hist']
max_depths = [3, 5, 7, 9, 11, 13, 15]
etas = [.01, .1]
gammas = [0, 1, 5]
colsample_bytrees = [.5, .7, 1]
n_models = len(boosters) * len(max_depths) * len(etas) * len(gammas) * len(colsample_bytrees)

start = time()
n = 0
for booster in boosters:
    # for tree_method in tree_methods:
    for max_depth in max_depths:
        for eta in etas:
            for gamma in gammas:
                for colsample_bytree in colsample_bytrees:
                    iteration = time()
                    cv_params = params.copy()
                    train_params = dict(max_depth=max_depth,
                                        eta=eta,
                                        gamma=gamma,
                                        colsample_bytree=colsample_bytree,
                                        # tree_method=tree_method,
                                        booster=booster)
                    cv_params.update(train_params)

                    cv_result = run_cv(test_params=cv_params,
                                       data=datasets,
                                       n_splits=n_splits)

                    for result, data in cv_result.items():
                        cv_results[result] = pd.concat([cv_results[result],
                                                        data.assign(model=n)])

                    out = f'\n\tModel: {n} of {n_models} | '
                    out += f'Iteration: {format_time(time() - iteration)} | '
                    print(out + f'Total: {format_time(time() - start)}\n')

                    cv_params['time'] = time() - iteration
                    models[n] = pd.Series(cv_params)
                    n += 1

cv_results['learning_curve'].columns = pd.MultiIndex.from_tuples(cv_results['learning_curve'].columns)
with pd.HDFStore('model_tuning.h5') as store:
    for result, data in cv_results.items():
        store.put(f'xgboost/{result}', data)
    store.put('xgboost/models', models)
