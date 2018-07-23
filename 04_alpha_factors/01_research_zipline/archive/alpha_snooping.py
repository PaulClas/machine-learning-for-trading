#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import warnings

warnings.filterwarnings('ignore')
import pandas as pd
from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens.performance import *
from alphalens.plotting import *
from os.path import join
from alphalens.tears import *
from utils import get_factor, get_quandl_wiki, get_sp500_fw
from pandas.tseries.offsets import BDay
from pprint import pprint

idx = pd.IndexSlice
pd.set_option('display.expand_frame_repr', False)

start = 2004
end = 2017
# periods = [1, 3, 5, 10, 20, 30]
periods = [10, 30, 60, 90, 120, 180, 240]
period_cols = [f'{p}D' for p in periods]
mavgs = [6, 12, 18, 24]


def get_benchmark(factor_data):
    factor_dates = factor_data.index.get_level_values('date').unique()
    last = (factor_dates.max() + BDay(max(periods))).year + 1
    return get_sp500_fw(start=start, end=last, periods=periods).reindex(factor_dates)


def eval_by_quantile(ma=6, duration=4):
    periods = [10, 30, 60, 90, 120, 180, 240]
    factor_cols = ['factor', 'factor_quantile']
    with pd.HDFStore('momentum_strategy.h5') as hdf:
        print(f'\n{ma} | {duration}\n')

        factor = hdf[f'factor/{ma}']
        factor_data = hdf[f'factor_data/{ma}']
    prices = get_quandl_wiki(stocks=factor.columns,
                             start=factor.index.min().year,
                             end=factor.index.max().year)

    metrics = pd.DataFrame()
    momentum_stocks = {}
    for holding_period in periods:
        print(f'{holding_period}', end=' ')
        shifted_period_returns = prices.pct_change(holding_period).shift(-holding_period).dropna(how='all', axis=1)
        grouped_ret = shifted_period_returns.groupby(pd.Grouper(freq=f'{duration}A'))

        corr = grouped_ret.apply(lambda x: x.corrwith(factor))
        corr = corr.apply(pd.qcut, q=5, labels=False, duplicates='drop', axis=1).dropna(1)

        return_quintiles = grouped_ret.mean().apply(pd.qcut, q=5, labels=False, duplicates='drop', axis=1)
        top_return_quintile = return_quintiles.filter(corr.columns).where(corr == 4).loc['2005':]

        for date, quintiles in top_return_quintile.iterrows():
            if not momentum_stocks.get(date.year):
                momentum_stocks[date.year] = {}
            momentum_stocks[date.year][holding_period] = quintiles.dropna().index.tolist()

    for end, periods in momentum_stocks.items():
        print(f'\n{end}\n\t', end=' ')
        factors = []
        start = end - duration + 1
        for holding_period, stocks in periods.items():
            print(f'{holding_period}', end=' ', flush=True)
            period_col = f'{holding_period}D'

            period_factor = factor_data.loc[idx[str(start): str(end), stocks], [period_col] + factor_cols]
            factors.append(period_factor.set_index(factor_cols, append=True))

            period_benchmark = get_benchmark(period_factor).loc[:, period_col]

            df = factor_alpha_beta(period_factor, returns=period_benchmark)
            df.loc['ic', period_col] = mean_information_coefficient(period_factor)[period_col]
            df = df.append(mean_return_by_quantile(period_factor)[0])
            df = df.T.reset_index().rename(columns={'index': 'period', 'Ann. alpha': 'alpha'}).assign(year=end)

            metrics = metrics.append(df)
        # with pd.HDFStore('momentum_strategy.h5') as hdf:
        #     hdf.put(f'period_factor/{ma}/{duration}/{end}', pd.concat(factors).reset_index(factor_cols).sort_index(1))
    # print(metrics.sort_values('alpha', ascending=False))
    return metrics.assign(ma=ma, duration=duration).sort_values('alpha', ascending=False)


# results = pd.DataFrame()
# for ma in mavgs:
#     for duration in [3, 4, 5]:
#         results = results.append(eval_by_q(ma=ma, duration=duration))

with pd.HDFStore('momentum_strategy.h5') as hdf:
    results = hdf.get('results')
cols = ['ma', 'period', 'duration', 'year']
results = results[results.year<2018]
print(results.sort_values('alpha', ascending=False).head(20))
print(results.info())
results.period = results.period.str.replace('D', '').astype(int)
print(results.groupby(['ma', 'year', 'period']).alpha.mean().unstack('period'))
# print(results.groupby(['ma', 'year', 'duration']).mean().sort_values('alpha', ascending=False))
exit()


def get_factor_data(start, end, quantiles, periods, ma_short, ma_long):
    factor = get_factor(start, end, ma_short, ma_long)
    stocks = factor.columns
    factor = factor.stack()
    factor.index.names = ['data', 'asset']

    prices = get_quandl_wiki(stocks=stocks,
                             start=start,
                             end=end)

    return get_clean_factor_and_forward_returns(factor=factor,
                                                prices=prices,
                                                quantiles=quantiles,
                                                periods=periods)


def get_data():
    for ma_short in [5, 10, 20]:
        print(ma_short)
        for ma_long in [25, 50, 100, 150]:
            print(ma_long)
            with pd.HDFStore('reversal_test.h5') as store:
                df = get_factor_data(start=start, end=end, quantiles=quantiles, periods=periods,
                                     ma_short=ma_short, ma_long=ma_long)
                store.put(f'factor_data/{ma_short}/{ma_long}', df)
                store.put(f'sp500/{ma_short}/{ma_long}', get_benchmark(df))


def eval_alpha(ma_short, ma_long):
    with pd.HDFStore('reversal_test.h5') as store:
        factor = store.get(f'factor_data/{ma_short}/{ma_long}')
        sp500 = store.get(f'sp500/{ma_short}/{ma_long}')

    first = sp500.index.min().year
    last = sp500.index.max().year

    idx = pd.IndexSlice
    ab, ic, ret, std = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for start in range(first, last):
        print(f'\t{start}')
        for end in range(start + 1, last + 1):
            s, e = str(start), str(end)
            p = {'start': start, 'end': end}
            f = factor.loc[idx[s: e, :], :]
            sp = sp500.loc[s: e, :]
            ab_ = factor_alpha_beta(f, returns=sp)
            ab = pd.concat([ab, ab_.assign(**p)])
            ic_ = mean_information_coefficient(f).to_frame().T
            ic = pd.concat([ic, ic_.assign(**p)], ignore_index=True)
            ret_, std_ = mean_return_by_quantile(f)
            ret = pd.concat([ret, ret_.assign(**p)])
            std = pd.concat([std, std_.assign(**p)])

    with pd.HDFStore('reversal_test.h5') as store:
        store.put(f'ic_period/{ma_short}/{ma_long}', ic)
        store.put(f'alpha_beta/{ma_short}/{ma_long}', ab)
        store.put(f'ret_q/{ma_short}/{ma_long}', ret)
        store.put(f'std_q/{ma_short}/{ma_long}', std)


# for ma_short in [20]:
#     print(ma_short)
#     for ma_long in [25, 50, 100, 150]:
#         print(ma_long)
#         eval_alpha(ma_short=ma_short, ma_long=ma_long)
#


def get_best_periods(ma):
    with pd.HDFStore('momentum_test.h5') as store:
        # params = dict(ma_short=ma_short, ma_long=ma_long)
        params = dict(ma=ma)
        # idx = ['ma_short', 'ma_long', 'start', 'end']
        idx = ['ma', 'start', 'end']
        # key = f'{ma_short}/{ma_long}'
        key = f'{ma}'
        ic = store.get(f'ic_period/{key}').assign(**params)
        ic = ic[ic.end - ic.start > 2].set_index(idx)
        ab = store.get(f'alpha_beta/{key}').assign(**params)
        ab = ab[ab.end - ab.start > 2]

    ab.index.names = ['metric']
    ab = ab.set_index(idx, append=True)
    res = pd.DataFrame()
    for period in ab.columns:
        ic_ = ic.loc[ic[period] > 0, period].to_frame('ic')
        ab_ = ab.loc[(ab.index.get_level_values('metric') == 'Ann. alpha') & (ab[period] > 0), period].to_frame('alpha')
        best = pd.concat([ic_, ab_.reset_index('metric', drop=True)], axis=1).dropna(how='all').assign(period=period)
        res = pd.concat([res, best])
    return res.dropna()


# results = pd.DataFrame()
# for ma_short in [5, 10, 20]:
#     print(ma_short)
#     for ma_long in [25, 50, 100, 150]:
#         print(f'\t{ma_long}')
#         results = pd.concat([results, get_best_periods(ma_short, ma_long)])
#
# print(results.sort_values('alpha', ascending=False))
# with pd.HDFStore('reversal_test.h5') as store:
#     store.put('best_strategies', results.sort_values('alpha', ascending=False))


# results = pd.DataFrame()
# for ma in [6, 12, 18, 24, 36]:
#     results = pd.concat([results, get_best_periods(ma)])
# print(results.sort_values('alpha', ascending=False))
# with pd.HDFStore('momentum_test.h5') as store:
#     store.put('best_strategies', results.sort_values('alpha', ascending=False))


def get_winners():
    with pd.HDFStore('momentum_test.h5') as store:
        # print(store.info())
        results = store.get('best_strategies')
        # print(results.groupby(level='ma').size())
        wins = pd.DataFrame()
        # for (ma_short, ma_long), data in results.groupby(level=['ma_short', 'ma_long']):
        for ma, data in results.groupby(level='ma'):
            # print(ma)
            ret = store.get(f'ret_q/{ma}').set_index(['start', 'end'], append=True)
            diff = ret.groupby(level=['start', 'end']).apply(lambda x: x.diff())
            diff = diff.where(diff > 0).dropna(how='all')
            candidates = diff.groupby(level=['start', 'end']).apply(lambda x: x.count() == 4)
            candidates = candidates.where(candidates).dropna(how='all').dropna(how='all', axis=1).reset_index()
            match = data.reset_index().merge(candidates).dropna(how='all', axis=1)
            if not match.empty:
                # print(match.head())
                cols = [c for c in match.columns if c.endswith('D')]
                # print(cols)
                wins = pd.concat([wins, match[match.period.isin(cols)].drop(cols, axis=1)])
        wins = wins.sort_values('alpha', ascending=False)
        print(wins)
        return wins


winners = get_winners()

"""
   ma  start   end        ic     alpha period
2   6   2012  2015  0.023413  0.021409   120D
6  12   2012  2015  0.026985  0.014392    10D
2  36   2011  2015  0.027431  0.010673    30D
2  18   2010  2015  0.026068  0.005114    90D
"""

with pd.HDFStore('momentum_test.h5') as store:
    for winner in range(3):
        w = winners.iloc[winner]
        ret = store[f'ret_q/{w.ma}']
        ret = ret.loc[(ret.start == w.start) & (ret.end == w.end)]
        print(ret)
