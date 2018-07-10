#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.expand_frame_repr', False)


def get_sp500_prices(start=2010, end=2017):
    with pd.HDFStore('assets.h5') as store:
        sp500 = store.get('sp500_constituents').ticker.tolist()
        asset_prices = store.get('wiki').adj_close.unstack().filter(sp500)
        return asset_prices.loc[str(start):str(end)].dropna(how='all', axis=1)


def get_wiki_prices(start=2010, end=2017):
    with pd.HDFStore('assets.h5') as store:
        asset_prices = store.get('wiki').adj_close.unstack().loc[str(start):str(end)]
        return asset_prices.dropna(thresh=int(len(asset_prices) * .8), axis=1)


df = get_wiki_prices(2002, 2017).pct_change().dropna(how='all').fillna(0)
print(df.info())
pca = PCA()
pca.fit(df)

print(pd.Series(pca.explained_variance_ratio_).head(10))
components = pca.components_
idx = np.argsort(components[:2, :])
pc1 = df.columns[idx[0, :100]]
pc2 = df.columns[idx[1, :100]]
print(len(pc1.difference(pc2)))
print(len(pc2.difference(pc1)))

