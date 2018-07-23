#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import empyrical as emp
import pyfolio as pf
import alphalens
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option('display.expand_frame_repr', False)

warnings.filterwarnings('ignore')

data_dir = Path('../data')


def get_asset_returns(start=2010, end=2017, outliers=1):
    with pd.HDFStore(str(data_dir / 'assets.h5')) as store:
        asset_prices = store.get('zip_quandl')
        asset_prices = asset_prices.loc[str(start):str(end)]
        outliers = asset_prices.pct_change().abs().gt(outliers).any()
        asset_prices = asset_prices.loc[:, outliers[~outliers].index]
        return asset_prices.pct_change().dropna(how='all')


def ma_crossover(start=2013, end=2017, outliers=1, ma1=20, ma2=200, hp_long=5, hp_short=2):
    asset_returns = get_asset_returns(start, end, outliers)
    start_date = asset_returns.index[0] + BDay(ma2)
    print('# Stocks:', asset_returns.shape[1], '\n')

    mav, entry, weights, returns = {}, {}, {}, {}
    for ma in [ma1, ma2]:
        mav[ma] = asset_returns.add(1).rolling(ma).apply(lambda x: x.prod() ** (-1 / len(x)) - 1)
        mav[ma] = mav[ma].dropna(how='all').loc[start_date:]

    strategies = [{'name': 'long', 'rule': 'gt', 'mult': 1, 'hp': hp_long},
                  {'name': 'short', 'rule': 'lt', 'mult': -1, 'hp': hp_short}]
    for s in strategies:
        entry[s['name']] = getattr(mav[ma1], s['rule'])(mav[ma2]).astype(int)
        weights[s['name']] = entry[s['name']].div(entry[s['name']].sum(axis=1), axis=0)
        weights[s['name']] = weights[s['name']].fillna(method='ffill', limit=s['hp']).fillna(0).mul(s['mult'])
        returns[s['name']] = weights[s['name']].shift().mul(asset_returns.loc[start_date:])

    returns['pf'] = returns['long'].add(returns['short'])
    plot_returns(returns)
    return {k: v.sum(axis=1).add(1).fillna(1).prod() - 1 for k, v in returns.items()}


def momentum(start=2013, end=2017, period=30, lower=.02, upper=.98):
    asset_returns = get_asset_returns(start, end)
    print('# Stocks:', asset_returns.shape[1], '\n')

    start_date = asset_returns.index[0] + pd.DateOffset(days=period)
    entry, weights, returns = {}, {}, {}

    rolling_ret = asset_returns.add(1).rolling(period).apply(lambda x: x.prod() ** (-1 / period) - 1)
    rolling_ret = rolling_ret.dropna(how='all').loc[start_date:]

    quantiles = rolling_ret.quantile(q=[lower, upper], axis=1).T
    strategies = [{'name': 'long', 'signal': 'gt', 'q': upper, 'fact': 1},
                  {'name': 'short', 'signal': 'lt', 'q': lower, 'fact': -1}]

    for strat in strategies:
        entry[strat['name']] = getattr(rolling_ret, strat['signal'])(quantiles[strat['q']], axis=0).astype(int)
        weights[strat['name']] = entry[strat['name']].div(entry[strat['name']].sum(axis=1), axis=0).mul(strat['fact'])
        returns[strat['name']] = weights[strat['name']].shift().mul(asset_returns.loc[start_date:]).dropna(how='all')

    returns['pf'] = returns['long'].add(returns['short'])
    # ret = returns['pf'].add(1).fillna(1).prod().sub(1).sort_values()
    # print(ret.head().append(ret.tail()))

    return {k: v.sum(axis=1).add(1).fillna(1).prod() - 1 for k, v in returns.items()}

# returns = get_quantiles_returns(lower=.05, upper=.95)
# print(pd.Series(returns))
# exit()

results = pd.DataFrame()
for outliers in [.5, 1]:
    for start in range(2010, 2016, 2):
        for period in range(10, 50, 10):
            for lower in [.01, .03, .05]:
                for upper in [.95, .97, .99]:
                    returns = get_quantiles_returns(outlers=outliers, start=start, end=2017, lower=lower, upper=upper)
                    returns['start'] = start
                    returns['period'] = period
                    returns['lower'] = lower
                    returns['upper'] = upper
                    returns['outliers'] = outliers
                    returns = pd.Series(returns)
                    print(returns)
                    results = pd.concat([results, returns.to_frame()], axis=1, ignore_index=True)

with pd.HDFStore('test.h5') as store:
    store.put('test_results', results.T)
