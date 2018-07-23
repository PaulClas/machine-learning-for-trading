#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import pandas as pd
import numpy as np
from numpy import isnan
from numpy.ma import count
from os.path import join
import statsmodels.api as sm
import warnings
from os.path import join
warnings.filterwarnings('ignore')
pd.set_option('display.expand_frame_repr', False)


def get_stock_prices(start=2010, end=2017):
    with pd.HDFStore('assets.h5') as store:
        sp500 = store.get('sp500_constituents').ticker.tolist()
        asset_prices = store.get('wiki').adj_close.unstack().filter(sp500)
        return asset_prices.loc[str(start):str(end)].dropna(how='all', axis=1)


def eval_assets():
    with pd.HDFStore('assets.h5') as store:
        sp500 = store.get('sp500_constituents').ticker
        df = store.get('quandl').filter(sp500).loc['2010':]

    print(df.info())
    annual_returns = df.groupby(pd.TimeGrouper('A')).apply(lambda x: x.iloc[-1].div(x.iloc[0]).sub(1))
    print(annual_returns.agg(['min', 'max']).T.sort_values('min').head(10))
    print(annual_returns.agg(['min', 'max']).T.sort_values('max', ascending=False).head(10))


with pd.HDFStore('alpha_factors.h5') as store:
    returns = get_stock_prices(2012, 2016)
    factor = store['momentum_factor'].loc['2012': '2016']
    quantiles = factor.apply(pd.qcut, q=20, labels=False, axis=1).stack().to_frame('quantile')
    data = pd.concat([factor.stack().to_frame('factor'), quantiles], axis=1)
    periods = [1, 3, 5, 10, 30]
    ic = pd.DataFrame()
    for period in periods:
        p = f'{period}D'
        df = pd.concat([data, returns.shift(-period).stack().to_frame(p)], axis=1).dropna()
        ic = pd.concat([ic, df.groupby('quantile')['factor', p].apply(lambda x: x.factor.corr(x[p])).to_frame(p)], axis=1)

    print(ic)