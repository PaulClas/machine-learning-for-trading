#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pprint import pprint
from pandas_datareader.famafrench import get_available_datasets
import pandas as pd
from statsmodels.api import OLS, add_constant
import pandas_datareader.data as web
from pathlib import Path
from linearmodels.asset_pricing import TradedFactorModel

# pprint(get_available_datasets())

data_path = Path('..', '..', 'data')

factors = 'F-F_Research_Data_5_Factors_2x3_daily'
ff = web.DataReader(factors, 'famafrench')[0]
# ff.iloc[:, 1:] = ff.iloc[:, 1:].sub(ff.iloc[:, 0], axis=0)
print(ff.corr())

with pd.HDFStore(data_path / 'assets.h5') as store:
    # stock_prices = store['quandl/wiki/prices'].close
    # stock_returns = stock_prices.unstack().pct_change().dropna(how='all').filter(ff.index, axis=0)
    sp500_returns = store['sp500/prices'].close.pct_change().filter(ff.index, axis=0).mul(100)
sp500_returns.name = 'SP500'

model = OLS(endog=sp500_returns, exog=add_constant(ff)).fit()
print(model.summary())

mod = TradedFactorModel(sp500_returns, ff)
res = mod.fit()
print(dir(res))
print(res.full_summary)
