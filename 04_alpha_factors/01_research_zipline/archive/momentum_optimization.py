#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import pandas as pd
import numpy as np
from numpy import isnan
from numpy.ma import count
from os.path import join
import warnings
from datetime import timedelta

warnings.filterwarnings('ignore')
pd.set_option('display.expand_frame_repr', False)


def rolling_ret(x):
    obs = count(x)
    x[isnan(x)] = 1
    return x.prod() ** (1 / obs) - 1


def compute_factor():
    data = get_stock_prices().pct_change()

    ma_short, ma_long = 20, 100
    ma1 = data.add(1).rolling(ma_short).apply(rolling_ret, raw=True)  # period return
    ma2 = data.add(1).rolling(ma_long).apply(rolling_ret, raw=True)  # period return

    momentum = ma1.sub(ma2).dropna(how='all')  # mavg to momentum

    # quantile = .95
    # quantiles = momentum.quantile(q=quantile, axis=1)  # mavg diff quantiles
    # entry = momentum.gt(quantiles, axis=0).astype(int).fillna(0)
    # print(entry.info())

    with pd.HDFStore('alpha_factor.h5') as store:
        store.put('transformed_data', momentum)

compute_factor()


def test_signal():
    with pd.HDFStore('alpha_factor.h5') as store:
        data = store.get('transformed_data')
        start, end = data.index[0], data.index[-1]

        stock_returns = get_stock_prices(start, end).pct_change()
        sp500_returns = get_sp500_returns(start, end)

        holding_period = 5
        max_pos = 0.05

        quantiles = data.quantile(q=quantile, axis=1)  # mavg diff quantiles
        entry = data.gt(quantiles, axis=0).astype(int).replace(0, np.nan)
        entry = entry.fillna(method='ffill', limit=holding_period)  # holding period
        weights = entry.div(entry.sum(axis=1).fillna(1), axis=0).clip(upper=max_pos).fillna(0)  # 1/n weight
        returns = weights.shift().mul(stock_returns).sum(1)  # position return
        print(returns.loc['2012': '2016'].to_frame('strategy').join(sp500_returns).add(1).prod().sub(1))


def compute_signal():
    with pd.HDFStore('momentum.h5') as store:
        data = store.get('momentum/20/100')
        start, end = data.index[0], data.index[-1]

        stock_returns = get_stock_prices(start, end).pct_change()
        sp500_returns = get_sp500_returns(start, end)

        lower, upper = 0.01, .99
        strategies = {'long' : {'rule': 'gt', 'leverage': 1, 'quantile': upper, 'hp': 1, 'max_pos': .05},
                      'short': {'rule': 'lt', 'leverage': -1, 'quantile': lower, 'hp': 1, 'max_pos': .05}}
        quantiles = data.quantile(q=[lower, upper], axis=1).T  # mavg diff quantiles
        quantiles[lower] = quantiles[lower].clip(upper=0)
        quantiles[upper] = quantiles[upper].clip(lower=0)
        mavg, entry, weights, returns = {}, {}, {}, pd.DataFrame()
        for strategy, params in strategies.items():
            entry = getattr(data, params['rule'])(quantiles[params['quantile']], axis=0)
            entry = entry.astype(int).replace(0, np.nan)  # entry by strategy
            entry = entry.fillna(method='ffill', limit=params['hp'])  # holding period
            weights = entry.div(entry.sum(axis=1).fillna(1), axis=0).clip(upper=params['max_pos'])  # 1/n weight
            weights = weights.fillna(0).mul(params['leverage'])  # long/short
            returns[strategy] = weights.shift().mul(stock_returns).sum(1)  # position return
            # print(returns[strategy].sum(1).add(1).prod() - 1)
        returns['sp500'] = sp500_returns

        print(returns.head())
        d = end.year - start.year
        baseline = sp500_returns.add(1).prod() ** (1 / d) - 1

        exit()

        print(f'sp500: {baseline:.2%}')

        quantiles = 10
        pos_signal = data.where(data > 0).apply(pd.qcut, q=quantiles, duplicates='drop', labels=False, axis=1)
        neg_signal = data.where(data > 0).apply(pd.qcut, q=quantiles, duplicates='drop', labels=False, axis=1)
        for signal in [pos_signal, neg_signal]:
            for q in range(quantiles):
                positions = (signal == q).fillna(False).astype(int)
                weights = positions.div(positions.sum(1), axis=0).shift().fillna(0)
                returns = weights.mul(stock_returns)
                ar = returns.sum(1).add(1).prod() ** (1 / d) - 1
                print(f'{q}: {ar:.2%}')
    #     short = (signal < 2).fillna(False).astype(int).mul(-1)
    stock_returns = get_stock_prices().pct_change()
    #     short = short.div(short.abs().sum(1), axis=0)
    #
    #     long = (signal > 17).fillna(False).astype(int)
    #     long = long.div(long.abs().sum(1), axis=0)
    #
    #     weights = short.add(long).shift().fillna(0)
    # with pd.HDFStore('momentum.h5') as store:
    #     store.put('ar_xs_vs_sp500_pos', weights)


def momentum(data, ma1=5, ma2=50, max_pos=.1):
    start_date = data.index[ma2]
    q = [.01, .03, .05, .1]
    holding_period = [0, 1, 3, 5, 10]
    strategies = {'long' : {'rule': 'gt', 'leverage': 1},
                  'short': {'rule': 'lt', 'leverage': -1}}

    mavg, entry, weights, returns = {}, {}, {}, {}
    mkey = join('momentum', str(ma1), str(ma2))
    with pd.HDFStore('momentum.h5') as store:
        keys = store.keys()
        if not any([k[1:] == mkey for k in keys]):
            for ma in [ma1, ma2]:
                mavg[ma] = data.add(1).rolling(ma).apply(rolling_ret, raw=True)  # period return
                mavg[ma] = mavg[ma].dropna(how='all').loc[start_date:]
            momentum = mavg[ma1].sub(mavg[ma2])  # mavg to momentum
            store.put(mkey, momentum)
        else:
            momentum = store.get(mkey)
        for lower in q:
            for upper in [1 - p for p in q]:
                qkey = join(mkey, str(int(lower * 100)), str(int(upper * 100)))
                if not any([k[1:] == qkey for k in keys]):
                    quantiles = momentum.quantile(q=[lower, upper], axis=1).T  # mavg diff quantiles
                    quantiles[lower] = quantiles[lower].clip(upper=0)
                    quantiles[upper] = quantiles[upper].clip(lower=0)
                    store.put(qkey, quantiles)
                else:
                    quantiles = store.get(qkey)
                for hp in holding_period:
                    kws = [str(i) for i in [ma1, ma2, int(lower * 100), int(upper * 100), hp]]
                    key = join('strategy', *kws)
                    if key not in keys:
                        strategies['long'].update({'q': upper, 'hp': hp})
                        strategies['short'].update({'q': lower, 'hp': hp})
                        for strategy, params in strategies.items():
                            entry[strategy] = getattr(momentum, params['rule'])(quantiles[params['q']], axis=0)
                            entry[strategy] = entry[strategy].astype(int).replace(0, np.nan)  # entry by strategy
                            if params['hp'] > 0:
                                entry[strategy] = entry[strategy].fillna(method='ffill',
                                                                         limit=params['hp'])  # holding period
                            weights[strategy] = entry[strategy].div(entry[strategy].sum(axis=1).fillna(1), axis=0).clip(
                                    upper=max_pos)  # 1/n weight
                            weights[strategy] = weights[strategy].fillna(0).mul(params['leverage'])  # long/short
                            returns[strategy] = weights[strategy].shift().mul(data.loc[start_date:])  # position return
                        result = pd.DataFrame({k: v.sum(axis=1) for k, v in returns.items()})
                        total = result.add(1).fillna(1).prod().sub(1)
                        print('\t|\t'.join(kws), f'\t|\t{total.long:.2%}\t|\t{total.short:.2%}')
                        store.put(key, result)
                    else:
                        continue


def param_opt():
    data = get_asset_returns(start=2010, end=2017)
    print('# Stocks:', data.shape, '\n')
    for ma1 in range(5, 21, 5):
        for ma2 in range(50, 201, 50):
            momentum(data, ma1=ma1, ma2=ma2)


def eval_returns():
    with pd.HDFStore('assets.h5') as store:
        sp500 = store.get('sp500').Close.pct_change().to_frame('sp500')
    # sp500.index = sp500.index.tz_localize('UTC')
    param_keys = ['ma1', 'ma2', 'lower', 'upper', 'hp']
    best_results = pd.DataFrame()
    with pd.HDFStore('momentum.h5') as store:
        strategies = [s for s in store.keys() if s[1:].startswith('strat')]
        for strategy in strategies:
            params = dict(zip(param_keys, [int(s) for s in strategy.split('/')[2:]]))
            assert param_keys == list(params.keys())
            returns = store[strategy].join(sp500).loc['2003':].dropna()
            monthly_returns = returns.groupby(pd.TimeGrouper('M')).agg(lambda x: x.add(1).prod().sub(1))
            for T in range(1, 6):
                params['T'] = T
                period_returns = monthly_returns.rolling(T * 12).apply(lambda x: np.prod(x + 1) - 1).dropna()
                period_returns.loc[:, ['long', 'short']] = period_returns.loc[:, ['long', 'short']].sub(
                        period_returns.sp500, axis=0)
                period_returns = period_returns.assign(**params)
                store.append('returns', period_returns, format='t')
                best_results = pd.concat([best_results,
                                          period_returns.nlargest(n=1, columns='long'),
                                          period_returns.nlargest(n=1, columns='short')])

        print(best_results.sort_values('long', ascending=False).head(10))
        print(best_results.sort_values('short', ascending=False).head(10))
        store.put('returns_top', best_results)


def test_signal_alt(start=2012, end=2016):
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
