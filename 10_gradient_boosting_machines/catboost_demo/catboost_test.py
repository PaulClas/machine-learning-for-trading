#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from time import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings('ignore')
pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)


class OneStepTimeSeriesSplit:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the index contains a level labeled 'date'"""

    def __init__(self, n_splits=3, test_period_length=1):
        self.n_splits = n_splits
        self.test_period_length = test_period_length

    @staticmethod
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def split(self, X, y=None, groups=None):
        unique_dates = (X.index.
                            get_level_values('date')
                            .unique()
                            .sort_values(ascending=False)
        [:self.n_splits * self.test_period_length])

        dates = X.reset_index()[['date']]
        for test_date in self.chunks(unique_dates, self.test_period_length):
            train_idx = dates[dates.date < min(test_date)].index
            test_idx = dates[dates.date.isin(test_date)].index
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


with pd.HDFStore('data.h5') as store:
    data = store['data']

y = data.returns
y_binary = (y > 0).astype(int)
X = data.drop('returns', axis=1)

X = X.reset_index('ticker')
X.ticker = pd.factorize(X.ticker)[0]
X.sector = pd.factorize(X.sector)[0]
cat_cols = ['year', 'month', 'sector', 'age', 'size']
X.loc[:, cat_cols] = X.loc[:, cat_cols].fillna(-1)
# print(X.info())

cv = OneStepTimeSeriesSplit(n_splits=10, test_period_length=1)

cat_clf = CatBoostClassifier(iterations=500,
                             learning_rate=0.03,
                             depth=6,
                             l2_leaf_reg=3,
                             model_size_reg=None,
                             rsm=None,
                             loss_function='Logloss',
                             border_count=32,
                             feature_border_type='MinEntropy',
                             od_pval=None,
                             od_wait=None,
                             od_type=None,
                             nan_mode='Min',
                             counter_calc_method=None,  #
                             leaf_estimation_iterations=1,
                             leaf_estimation_method='Gradient',
                             thread_count=None,
                             random_seed=None,
                             use_best_model=None,
                             best_model_min_trees=None,
                             verbose=True,
                             logging_level=None,
                             metric_period=100,
                             simple_ctr=None,
                             ctr_leaf_count_limit=None,
                             store_all_simple_ctr=None,
                             max_ctr_complexity=1,
                             gpu_ram_part=0.2,
                             has_time=None,
                             allow_const_label=None,
                             classes_count=None,
                             class_weights=None,
                             one_hot_max_size=None,
                             random_strength=None,
                             name=None,
                             ignored_features=None,
                             train_dir=None,
                             custom_loss=None,
                             custom_metric=None,
                             eval_metric='AUC',
                             bagging_temperature=None,
                             save_snapshot=None,
                             snapshot_file=None,
                             snapshot_interval=None,
                             fold_len_multiplier=None,
                             used_ram_limit=None,
                             pinned_memory_size=None,
                             allow_writing_files=None,
                             final_ctr_computation_mode=None,
                             approx_on_full_history=None,
                             boosting_type='Ordered',
                             combinations_ctr=None,
                             per_feature_ctr=None,
                             ctr_description=None,
                             task_type='GPU',
                             bootstrap_type='Bayesian',
                             subsample=None,
                             dev_score_calc_obj_block_size=None,
                             gpu_cat_features_storage=None,
                             data_partition=None,
                             metadata=None,
                             cat_features=None)

start = time()
cat_auc = cross_val_score(cat_clf,
                          X=X,
                          y=y_binary,
                          cv=cv,
                          n_jobs=-1,
                          scoring='roc_auc',
                          verbose=2,
                          fit_params={'cat_features': [0] + list(range(24, 29))}
                          )
print(f'\nTook {time()-start:.2f}s')
print(cat_auc)
print(np.mean(cat_auc))
