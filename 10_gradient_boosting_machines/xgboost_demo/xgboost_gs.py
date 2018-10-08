#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import warnings
from random import shuffle
from time import time
import xgboost as xgb
from xgboost.callback import reset_learning_rate
from itertools import product
from gbm_utils import *
from gbm_params import get_params

warnings.filterwarnings('ignore')
idx = pd.IndexSlice
np.random.seed(42)


def learning_rate(n, ntot):
    start_eta = 0.1
    k = 8 / ntot
    x0 = ntot / 1.8
    return start_eta * (1 - 1 / (1 + np.exp(-k * (n - x0))))


def get_datasets(features, target, kfold):
    data = {}
    for fold, (train_idx, test_idx) in enumerate(kfold.split(features)):
        print(fold, end=' ', flush=True)
        data[fold] = {'train': xgb.DMatrix(label=target.iloc[train_idx],
                                           data=features.iloc[train_idx],
                                           nthread=-1),
                      'valid': xgb.DMatrix(label=target.iloc[test_idx],
                                           data=features.iloc[test_idx],
                                           nthread=-1)}
    return data


def run_cv(test_params, data, n_splits=10):
    """Train-Validate with early stopping"""
    result = []
    for fold in range(n_splits):
        train = data[fold]['train']
        valid = data[fold]['valid']

        scores = {}
        model = xgb.train(params=test_params,
                          dtrain=train,
                          evals=list(zip([train, valid], ['train', 'valid'])),
                          verbose_eval=50,
                          num_boost_round=250,
                          early_stopping_rounds=25,
                          evals_result=scores,
                          callbacks=[reset_learning_rate(learning_rate)])

        result.append([model.best_iteration,
                       scores['train']['auc'][-1],
                       scores['valid']['auc'][-1]])

    return (pd.DataFrame(result,
                         columns=['rounds', 'train', 'valid'])
            .mean()
            .append(pd.Series(test_params)))


model = 'xgboost'
y, features = get_data()
X = get_one_hot_data(features)
y, X, y_test, X_test = get_holdout_set(target=y,
                                       features=X)

with pd.HDFStore('model_tuning.h5') as store:
    store.put('xgb/holdout/features', X_test)
    store.put('xgb/holdout/target', y_test)

n_splits = 2
cv = OneStepTimeSeriesSplit(n_splits=n_splits)

datasets = get_datasets(features=X, target=y, kfold=cv, model=model)
results = pd.DataFrame()

param_grid = dict(
        booster=['gbtree', 'dart'],
        max_depth=list(range(3, 14, 2)),
        eta=[.1, .1, .3],
        gamma=[0, 1],
        colsample_bytree=[.6, .8, 1]
)

all_params = list(product(*param_grid.values()))
n_models = len(all_params)
shuffle(all_params)

start = time()
for n, test_param in enumerate(all_params, 1):
    iteration = time()

    cv_params = get_params(model)
    cv_params.update(dict(zip(param_grid.keys(), test_param)))

    results[n] = run_cv(test_params=cv_params,
                        data=datasets,
                        n_splits=n_splits)
    results.loc['time', n] = time() - iteration

    if n > 1:
        df = results[~results.eq(results.iloc[:, 0], axis=0).all(1)].T
        if 'valid' in df.columns:
            df.valid = pd.to_numeric(df.valid)
            print('\n')
            print(df.sort_values('valid', ascending=False).head(3).reset_index(drop=True))

    out = f'\n\tModel: {n} of {n_models} | '
    out += f'{format_time(time() - iteration)} | '
    out += f'Total: {format_time(time() - start)} | '
    print(out + f'Remaining: {format_time((time() - start)/n*(n_models-n))}\n')

    with pd.HDFStore('model_tuning.h5') as store:
        store.put('xgb/results', results.T.apply(pd.to_numeric, errors='ignore'))
