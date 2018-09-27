#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)

tree = dict(
        criterion='gini',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        class_weight=None,
        presort=False)

kwargs = pd.Series(tree).to_frame().reset_index()
kwargs.columns = ['keyword', 'default']

rf = dict(n_estimators=10,
          criterion='gini',
          max_depth=None,
          min_samples_split=2,
          min_samples_leaf=1,
          min_weight_fraction_leaf=0.0,
          max_features='auto',
          max_leaf_nodes=None,
          min_impurity_decrease=0.0,
          min_impurity_split=None,
          bootstrap=True,
          oob_score=False,
          n_jobs=1,
          random_state=None,
          verbose=0,
          warm_start=False,
          class_weight=None)

kwargs2 = pd.Series(rf).to_frame().reset_index()
kwargs2.columns = ['keyword', 'default']
kwargs = kwargs.assign(model='tree').append(kwargs2.assign(model='rf'))
kwargs.sort_values(['keyword', 'model']).to_csv('kwargs.csv', index=False)
