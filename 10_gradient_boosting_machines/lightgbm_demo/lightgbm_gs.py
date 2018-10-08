#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import warnings
from random import shuffle
from time import time
import lightgbm as lgb
from itertools import product
from .gs_utils import *

warnings.filterwarnings('ignore')
idx = pd.IndexSlice
np.random.seed(42)


def get_datasets(features, target, kfold):
    data = {}
    for fold, (train_idx, test_idx) in enumerate(kfold.split(features)):
        train = lgb.Dataset(label=target.iloc[train_idx],
                            data=features.iloc[train_idx],
                            categorical_feature=cat_cols,
                            free_raw_data=False)

        # align validation set histograms with training set
        valid = train.create_valid(label=target.iloc[test_idx],
                                   data=features.iloc[test_idx])

        data[fold] = {'train': train.construct(),
                      'valid': valid.construct()}
    return data


def run_cv(test_params, data, n_splits=10):
    """Train-Validate with early stopping"""
    result = []
    for fold in range(n_splits):
        train = data[fold]['train']
        valid = data[fold]['valid']

        scores = {}
        model = lgb.train(params=test_params,
                          train_set=train,
                          valid_sets=[train, valid],
                          valid_names=['train', 'valid'],
                          verbose_eval=50,
                          evals_result=scores)

        result.append([model.current_iteration(),
                       scores['train']['auc'][-1],
                       scores['valid']['auc'][-1]])

    return (pd.DataFrame(result,
                         columns=['rounds', 'train', 'valid'])
            .mean()
            .append(pd.Series(test_params)))


y, features = get_data()
X_factors = factorize_cats(features)
y, X_factors, y_test, X_test = get_holdout_set(target=y,
                                               features=X_factors)

with pd.HDFStore('model_tuning.h5') as store:
    store.put('lgb/holdout/features', X_test)
    store.put('lgb/holdout/target', y_test)

n_splits = 10
cv = OneStepTimeSeriesSplit(n_splits=n_splits)

datasets = get_datasets(features=X_factors, target=y, kfold=cv)
results = pd.DataFrame()

param_grid = dict(
        num_leaves=[2 ** i for i in [3, 4, 5]],
        min_data_in_leave=[20, 100, 500],
        colsample_bytree=[.5, .75, 1],
        eta=[.01, .1],
        is_unbalance=[True, False]
)

all_params = list(product(*param_grid.values()))
n_models = len(all_params)
shuffle(all_params)

start = time()
for n, test_param in enumerate(all_params, 1):
    iteration = time()

    cv_params = params.copy()
    cv_params.update(dict(zip(param_grid.keys(), test_param)))

    results[n] = run_cv(test_params=cv_params,
                        data=datasets,
                        n_splits=n_splits)
    results.loc['time', n] = time() - iteration

    if n > 1:
        df = results[~results.eq(results.iloc[:, 0], axis=0).all(1)].T
        df.valid = pd.to_numeric(df.valid)
        print('\n')
        print(df.sort_values('valid', ascending=False).head(3).reset_index(drop=True))

    out = f'\n\tModel: {n} of {n_models} | '
    out += f'{format_time(time() - iteration)} | '
    out += f'Total: {format_time(time() - start)} | '
    print(out + f'Remaining: {format_time((time() - start)/n*(n_models-n))}\n')

results = results.T.apply(pd.to_numeric, errors='ignore')
with pd.HDFStore('model_tuning.h5') as store:
    store.put('lgb/results', results)
