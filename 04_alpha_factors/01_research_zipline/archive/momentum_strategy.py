#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'
import pandas as pd
import numpy as np
from numpy import isnan
from numpy.ma import count
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.expand_frame_repr', False)


def get_sp500_prices(start=2010, end=2017):
    with pd.HDFStore('assets.h5') as store:
        sp500 = store.get('sp500_constituents').ticker.tolist()
        asset_prices = store.get('wiki').adj_close.unstack().filter(sp500)
        return asset_prices.loc[str(start):str(end)].dropna(how='all', axis=1)


def get_wiki_prices(start=2010, end=2017, na_thresh=.8):
    with pd.HDFStore('assets.h5') as store:
        asset_prices = store.get('wiki').adj_close.unstack().loc[str(start):str(end)]
        return asset_prices.dropna(thresh=int(len(asset_prices) * na_thresh), axis=1)


def get_sp500_returns(start=2010, end=2017):
    with pd.HDFStore('assets.h5') as store:
        sp500 = store['sp500'].Close.loc[str(start):str(end)]
        return sp500.pct_change().to_frame('sp500')


def rolling_ret(x):
    obs = count(x)
    x[isnan(x)] = 1
    return x.prod() ** (1 / obs) - 1


def momentum_12_2(ma=3, shift=0, start=2003, end=2017):
    prices = get_wiki_prices(start, end)
    returns = prices.pct_change().dropna(how='all')
    key = f'{ma}_{shift}'
    test_key = '/test_l_' + key
    with pd.HDFStore('momentum_12_2.h5') as store:
        if test_key in store.keys():
            momentum = store.get(test_key)
        else:
            momentum = returns.add(1).rolling(ma * 252).apply(rolling_ret)
            store.put(test_key, momentum)

    returns = returns.loc[momentum.index]
    corrs = {}
    for p in range(3, 25, 3):
        print(p, end=' ')
        T = p * 21
        outcome = returns.add(1).fillna(1).rolling(T).apply(lambda x: x.prod(), raw=True).pow(1 / T).sub(1)
        corrs[p] = momentum.corrwith(outcome.shift(-T))

    corrs = pd.DataFrame(corrs)
    print(corrs.describe())
    corr_key = '/corr_l_' + key
    with pd.HDFStore('momentum_12_2.h5') as store:
        store.put(corr_key, corrs)


momentum_12_2()
exit()


def test_factor(shift=20, ma=250, start=2010, end=2017):
    prices = get_wiki_prices(start, end)
    returns = prices.pct_change().dropna(how='all')
    # key = f'/test_{ma_short}_{ma_long}'
    key = f'{ma}_{shift}'
    test_key = '/test_' + key
    with pd.HDFStore('momentum_12_2.h5') as store:
        try:
            store.remove(test_key)
        except:
            pass
        if test_key not in store.keys():
            # ma1 = returns.add(1).rolling(ma_short).apply(rolling_ret, raw=True)  # period return
            # ma2 = returns.add(1).rolling(ma_long).apply(rolling_ret, raw=True)  # period return
            # momentum_factor = ma1.sub(ma2).dropna(how='all')  # mavg to momentum
            momentum_factor = returns.add(1).rolling(ma).apply(rolling_ret, raw=True).rank(1)
            momentum_factor = momentum_factor.shift(shift)
            store.put(test_key, momentum_factor)
        else:
            momentum_factor = store[test_key]

        corr_key = '/corr_' + key
        try:
            store.remove(corr_key)
        except:
            pass
        if corr_key not in store.keys():
            returns = returns.loc[momentum_factor.index]
            corrs = {}
            for p in [1, 5, 10, 30, 90, 180]:
                outcome = returns.add(1).fillna(1).rolling(p).apply(lambda x: x.prod(), raw=True).pow(1 / p).sub(1)
                corrs[p] = momentum_factor.corrwith(outcome.shift(-p))

            corrs = pd.DataFrame(corrs)
            print(corrs.describe())
            store.put(corr_key, corrs)


test_factor(ma=220, shift=20)
exit()

# for ma_short in [5, 10, 20]:
#     print(f'ma_short: {ma_short}')
#         print(f'\tma_long: {ma_long}')
#         test_factor(ma_short, ma_long)

exit()


def compute_factor():
    data = get_stock_prices().pct_change()

    ma_short, ma_long = 20, 100
    ma1 = data.add(1).rolling(ma_short).apply(rolling_ret, raw=True)  # period return
    ma2 = data.add(1).rolling(ma_long).apply(rolling_ret, raw=True)  # period return
    momentum_factor = ma1.sub(ma2).dropna(how='all')  # mavg to momentum
    print(ma1.tail())
    print(ma2.tail())
    print(momentum_factor.tail())
    exit()
    with pd.HDFStore('alpha_factors.h5') as store:
        store.put('momentum_factor', momentum_factor)


compute_factor()


def test_signal(start=2012, end=2016):
    with pd.HDFStore('alpha_factors.h5') as store:
        factor = store.get('momentum_factor').loc[str(start):str(end)]
        start, end = factor.index[0], factor.index[-1]
        d = (end - start).days

        stock_returns = get_stock_prices(start, end).pct_change()

        holding_period = 5
        max_pos = 0.05
        quantile = .95

        # identify factor signals
        quantiles = factor.quantile(q=quantile, axis=1)
        entry = factor.gt(quantiles, axis=0).astype(int).replace(0, np.nan)

        # remain invested for holding_period
        entry = entry.fillna(method='ffill', limit=holding_period)

        # set position to 1/n restricted to max_pos
        weights = entry.div(entry.sum(axis=1).fillna(1), axis=0).clip(upper=max_pos).fillna(0)

        # compute weighted position returns using next-day returns
        weighted_returns = weights.shift().mul(stock_returns)
        store.put('reversal_returns_weighted', weighted_returns)

        # compute portfolio returns
        pf_returns = weighted_returns.sum(1).to_frame('strategy')

        # compare against S&P 500
        sp500_returns = get_sp500_returns(start, end)

        returns = pf_returns.join(sp500_returns)
        store.put('reversal__returns', returns)
        print(returns.add(1).prod().pow(1 / d).pow(252).sub(1))


# compute_factor()
test_signal()
exit()
