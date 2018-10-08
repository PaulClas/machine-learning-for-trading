#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import warnings
from pprint import pprint
from time import time
from random import shuffle
from itertools import product
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score
from gbm_utils import *
from gbm_params import get_params

warnings.filterwarnings('ignore')
idx = pd.IndexSlice
np.random.seed(42)


def get_datasets(features, target, kfold):
    s = pd.Series(features.columns.tolist())
    cat_cols_idx = s[s.isin(cat_cols)].index.tolist()

    data = {}
    for fold, (train_idx, test_idx) in enumerate(kfold.split(features)):
        print(fold, end=' ', flush=True)
        train = Pool(label=target.iloc[train_idx],
                     data=features.iloc[train_idx],
                     cat_features=cat_cols_idx)

        valid = Pool(label=target.iloc[test_idx],
                     data=features.iloc[test_idx],
                     cat_features=cat_cols_idx)

        data[fold] = {'train': train,
                      'valid': valid}
    print()
    return data


def run_cv(test_params, data, n_splits=10):
    """Train-Validate with early stopping"""
    result = []
    model = CatBoostClassifier(**test_params)
    for fold in range(n_splits):
        print(fold, end=' ', flush=True)
        train = data[fold]['train']
        valid = data[fold]['valid']

        model.fit(X=train,
                  eval_set=[valid],
                  logging_level='Silent')

        train_score = model.predict_proba(train)[:, 1]
        valid_score = model.predict_proba(valid)[:, 1]
        result.append([
            model.tree_count_,
            roc_auc_score(y_score=train_score, y_true=train.get_label()),
            roc_auc_score(y_score=valid_score, y_true=valid.get_label())
        ])

    return (pd.DataFrame(result,
                         columns=['rounds', 'train', 'valid'])
            .mean()
            .append(pd.Series(test_params)))


y, features = get_data()
X_factors = factorize_cats(features)
y, X_factors, y_test, X_test = get_holdout_set(target=y,
                                               features=X_factors)


with pd.HDFStore('model_tuning.h5') as store:
    store.put('cat/holdout/features', X_test)
    store.put('cat/holdout/target', y_test)

n_splits = 12
cv = OneStepTimeSeriesSplit(n_splits=n_splits)

datasets = get_datasets(features=X_factors, target=y, kfold=cv)
results = pd.DataFrame()

param_grid = dict(
        depth=list(range(3, 17, 2)),
)

model = 'catboost'
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
        df.valid = pd.to_numeric(df.valid)
        print('\n')
        print(df.sort_values('valid', ascending=False).head(3).reset_index(drop=True))

    out = f'\n\n\tModel: {n} of {n_models} | '
    out += f'{format_time(time() - iteration)} | '
    out += f'Total: {format_time(time() - start)} | '
    print(out + f'Remaining: {format_time((time() - start)/n*(n_models-n))}\n')

results = results.T.apply(pd.to_numeric, errors='ignore')
with pd.HDFStore('model_tuning.h5') as store:
    store.put('cat/results', results)
