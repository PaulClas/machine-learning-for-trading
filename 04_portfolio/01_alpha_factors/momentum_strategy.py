#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import pandas as pd
import numpy as np
import warnings
from alphalens.performance import factor_information_coefficient, mean_information_coefficient, create_pyfolio_input, \
    mean_return_by_quantile
from alphalens.utils import get_clean_factor_and_forward_returns
from scipy.stats import spearmanr
from pathlib import Path
import pandas_market_calendars as mcal

np.random.seed(42)

warnings.filterwarnings('ignore')
pd.set_option('display.expand_frame_repr', False)

start = 1990
end = 2017
TRADING_DAYS = 21

idx = pd.IndexSlice

# holding_periods = [m * TRADING_DAYS for m in [3, 6, 9, 12]]
# holding_periods = [3, 6, 9, 12]
holding_periods = [1, 3, 5, 10, 20]
period_cols = [f'{p}D' for p in holding_periods]
mavgs = [6, 12, 18, 24]

ASSETS_STORE = str(Path('..', '00_data', 'assets.h5'))


def get_wiki_sample(start=None, end=None, col='adj_close'):
    df = pd.read_hdf(ASSETS_STORE, 'quandl/wiki/prices')
    df = df[col].unstack()

    if start is None:
        start = df.index[0]
    if end is None:
        end = df.index[-1]

    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=str(start), end_date=str(end + 1))
    trading_dates = mcal.date_range(schedule, frequency='1D').date

    df = df.loc[trading_dates].dropna(how='all')
    df.index = pd.to_datetime(df.index)
    df.columns.names = ['assets']
    return df


def information_coefficient(asset_prices: pd.DataFrame,
                            factor: pd.DataFrame,
                            holding_period: int = 12) -> pd.DataFrame:
    """Compute the Information Coefficient

    :param asset_prices: wide format DataFrame with DateTimeIndex; shape T x # assets
    :param factor: wide format DataFrame with DateTimeIndex; shape T x # assets
    :param holding_period: holding period
    :return: Information Period for each period
    """
    window = holding_period * TRADING_DAYS
    returns = asset_prices.ffill().rolling(window).apply(lambda x: (x[-1] / x[0]) - 1).shift(-window)
    returns = returns.where(asset_prices.notnull())
    common_dates = returns.dropna(how='all').index.intersection(factor.dropna(how='all').index)
    corr, p = spearmanr(a=returns.loc[common_dates], b=factor.loc[common_dates], axis=1, nan_policy='omit')
    return pd.DataFrame({'rho'    : np.diagonal(corr, offset=len(common_dates)),
                         'p-value': np.diagonal(p, offset=len(common_dates))}, index=common_dates)


def relative_momentum(asset_prices: pd.DataFrame,
                      lookback: int = 12,
                      shift: int = 0) -> pd.DataFrame:
    """Compute rolling period returns.

    :param asset_prices: DataFrame with prices in columns and DateTimeIndex
    :param lookback: # of months in period (default 12)
    :param shift: # of months to shift period returns
    :param quantiles: # quantiles for period return signal
    :return: pd.DataFrame
    """
    T = TRADING_DAYS * lookback
    S = TRADING_DAYS * shift
    momentum = asset_prices.pct_change(T, limit=int(T / 2))
    momentum = momentum.shift(S)
    return momentum.dropna(how='all', axis=1).dropna(how='all')


def momentum_overlay(asset_prices: pd.DataFrame,
                     relative_lookback: int = 12,
                     absolute_lookback: int = 3,
                     quantiles: int = 5) -> pd.DataFrame:
    limit = int(relative_lookback / 2)  # non-missing data per window
    relative_momemtum = asset_prices.pct_change(relative_lookback, limit=limit)
    relative_momemtum = relative_momemtum.dropna(how='all', axis=1).dropna(how='all')

    labels = list(range(1, quantiles + 1))
    relative_momemtum_quantiles = relative_momemtum.apply(pd.qcut,
                                                          q=quantiles,
                                                          labels=labels,
                                                          axis=1)

    absolute_momentum = asset_prices.pct_change(absolute_lookback)

    short = relative_momemtum.where((relative_momemtum_quantiles == 1) &
                                    (absolute_momentum < 0))

    long = relative_momemtum.where((relative_momemtum_quantiles == quantiles) &
                                   (absolute_momentum > 0))

    return short.combine_first(long).dropna(how='all', axis=1)


def daily_return(prices: pd.DataFrame, t: int) -> pd.DataFrame:
    """Compute daily returns for prices during period t"""
    return prices.pct_change(t).add(1).pow(1 / t).sub(1)


def mean_reversal(asset_prices: pd.DataFrame,
                  short_term: int = 10,
                  long_term: int = 50) -> pd.DataFrame:
    """Compute rolling period returns.

    :param asset_prices: DataFrame with prices in columns and DateTimeIndex
    :param lookback: # of months in period (default 12)
    :param shift: # of months to shift period returns
    :param quantiles: # quantiles for period return signal
    :return: pd.DataFrame
    """
    ma1 = daily_return(asset_prices, short_term)
    ma2 = daily_return(asset_prices, long_term)
    cross = ma1.sub(ma2, axis=0)
    short = cross.where((cross < 0) & (ma1 < 0))
    long = cross.where((cross > 0) & (ma1 > 0))
    return long.combine_first(short).mul(-1)


def opt_cross_ma(which='sp500'):
    print(which)
    close = get_wiki_sample(start=1988, end=2017, col='adj_close')
    open = get_wiki_sample(start=1988, end=2017, col='adj_open')

    if which == 'sp500':
        with pd.HDFStore(ASSETS_STORE) as store:
            sp500_stocks = store['sp500/stocks'].index.tolist()
        close = close.filter(sp500_stocks)
        open = open.filter(sp500_stocks)
    short_term = 10
    long_term = 50
    factor = mean_reversal(asset_prices=close, short_term=short_term, long_term=long_term)
    factor_data = get_clean_factor_and_forward_returns(factor=factor.stack(),
                                                       prices=open.shift(-1),
                                                       periods=holding_periods,
                                                       quantiles=4)
    with pd.HDFStore('momentum_factor.h5') as store:
        store.put(f'crossing_ma/{short_term}/{long_term}/{which}/factor_data', factor_data)


# opt_cross_ma()
# opt_cross_ma(which='all')

short_term = 10
long_term = 50
for which in ['sp500', 'all']:
    print(which)
    with pd.HDFStore('momentum_factor.h5') as store:
        factor_data = store[f'mean_reversal/{short_term}/{long_term}/{which}/factor_data'].loc[idx['2002':'2017', :], :]
    qmin, qmax = factor_data.factor_quantile.min(), factor_data.factor_quantile.max()

    input_data = create_pyfolio_input(factor_data,              # generated using alphalens
                                      period='1D',              # holding period
                                      capital=100000,           # starting capital
                                      long_short=False,         # market-neutral pf if True
                                      group_neutral=False,
                                      equal_weight=False,
                                      quantiles=[qmin, qmax],
                                      groups=None,
                                      benchmark_period='1D')
    returns, positions, benchmark = input_data

    key = f'mean_reversal/{short_term}/{long_term}/{which}/'
    with pd.HDFStore('../02_risk_metrics/risk_metrics.h5') as store:
        store.put(key + 'returns', returns)
        store.put(key + 'positions', positions)
        store.put(key + 'benchmark', benchmark)

exit()


def opt_params(prices):
    periods = months * TRADING_DAYS
    quantiles = 5
    factor = relative_momentum(prices, lookback=months, shift=1)
    factor = momentum_overlay(prices,
                              relative_lookback=periods,
                              absolute_lookback=int(periods / 2),
                              quantiles=quantiles)

    # positions = pd.concat([factor.lt(0).sum(1).to_frame('short'),
    #                        factor.gt(0).sum(1).to_frame('long')], axis=1)
    # print(positions.tail())
    # print(positions.resample('A').mean())

    print(factor_data.groupby(pd.Grouper(level='date', freq='A')).apply(get_returns))
    ic = factor_information_coefficient(factor_data)
    print(ic.resample('A').mean())
    with pd.HDFStore('momentum_factor.h5') as store:
        store.put(f'momentum_overlay/ic/{months}/sp500', ic)
        store.put(f'momentum_overlay/factor_data/{months}/sp500', factor_data)
        store.put(f'momentum_overlay/factor/{months}/sp500', factor)


def get_returns(x):
    return mean_return_by_quantile(x)[0]


def get_spear(x):
    a = x['factor'].dropna(how='all', axis=1)
    b = x['126D'].dropna(how='all', axis=1)
    print(x.shape, a.shape, b.shape)
    c, std = spearmanr(a=a, b=b, nan_policy='omit')
    return c


def get_candidates():
    with pd.HDFStore('momentum_factor.h5') as store:
        df = store.get(f'relative_momentum/factor_data/18').loc[:, ['126D', 'factor']]

    df = df.unstack().loc['2013': '2015'].dropna(how='all', axis=1)
    df = df.fillna(df.median())
    factor = df.loc[:, 'factor']

    fw = df.loc[:, '126D']
    print(fw.shape, factor.shape)
    stocks = factor.columns.tolist()

    corr, pval = spearmanr(a=factor, b=fw, axis=0)
    result = pd.DataFrame({'corr'   : np.diagonal(corr, offset=len(stocks)),
                           'p-value': np.diagonal(pval, offset=len(stocks))}, index=stocks)

    # corr = df.groupby(pd.Grouper(level='date', freq='A')).apply(get_spear)
    print(corr.nlargest(columns=['corr'], n=200))
    with pd.HDFStore('momentum_factor.h5') as store:
        store.put('relative_momentum/corr/18', result)


def opt_params(prices):
    for months in [6, 12, 18, 24, 36]:
        periods = months * TRADING_DAYS
        quantiles = 5
        # factor = relative_momentum(prices, lookback=months, shift=1)
        factor = momentum_overlay(prices,
                                  relative_lookback=periods,
                                  absolute_lookback=int(periods / 2),
                                  quantiles=quantiles)

        # positions = pd.concat([factor.lt(0).sum(1).to_frame('short'),
        #                        factor.gt(0).sum(1).to_frame('long')], axis=1)
        # print(positions.tail())
        # print(positions.resample('A').mean())
        factor_data = get_clean_factor_and_forward_returns(factor=factor.stack(),
                                                           prices=prices,
                                                           periods=holding_periods,
                                                           quantiles=quantiles)

        print(factor_data.groupby(pd.Grouper(level='date', freq='A')).apply(get_returns))
        ic = factor_information_coefficient(factor_data)
        print(ic.resample('A').mean())
        with pd.HDFStore('momentum_factor.h5') as store:
            store.put(f'momentum_overlay/ic/{months}/sp500', ic)
            store.put(f'momentum_overlay/factor_data/{months}/sp500', factor_data)
            store.put(f'momentum_overlay/factor/{months}/sp500', factor)


def optimize():
    with pd.HDFStore(ASSETS_STORE) as store:
        sp500_stocks = store['sp500/stocks'].index.tolist()

    close = get_wiki_sample(start=1988, end=2017)
    close = close.filter(sp500_stocks)

    print(close.info())
    opt_params(prices=close)


def pyfolio_data(asset_prices, start=2013, end=2015, q=5, p=periods):
    factor = get_momentum(asset_prices, months=24).loc[str(start): str(end)]
    factor = factor.stack()
    factor.index.names = ['date', 'asset']
    factor_data = get_clean_factor_and_forward_returns(factor, asset_prices,
                                                       periods=p,
                                                       quantiles=q)

    print(factor_data.info())
    return create_pyfolio_input(factor_data,
                                period='63D',
                                capital=1e6,
                                long_short=True,
                                group_neutral=False,
                                equal_weight=False,
                                quantiles=None,
                                groups=None,
                                benchmark_period='63D')


returns, positions, benchmark = pyfolio_data(asset_prices=prices)

print(returns.head())
print(positions.head())
print(positions.info())
print(benchmark)
print(benchmark.head())

with pd.HDFStore('momentum_factor.h5') as store:
    store.put('returns', returns)
    store.put('positions', positions)
    store.put('benchmark', benchmark)
exit()


def get_signal(factor, quantiles):
    quantiles = factor.apply(pd.qcut, q=quantiles, labels=False, axis=1)
    return quantiles.dropna(how='all')


def factor_ic(factor):
    start = factor.index[0]
    summary, total = pd.DataFrame(), pd.DataFrame()
    for p in [10, 30, 60, 90, 120, 180, 240]:
        outcome = prices.loc[start:].rolling(p).apply(lambda x: (x[-1] / x[0]) ** (1 / p) - 1)
        corr = factor.resample('A').apply(lambda x: x.corrwith(outcome.shift(-p))).T
        total[p] = corr.abs().median()
        summary = pd.concat([summary, corr.describe().iloc[1:].assign(p=p)])
    summary = summary.reset_index().sort_values(['index', 'p'])
    return summary, total


def tune_params():
    for ma in [6, 12, 18, 24, 36]:
        print(ma)
        factor = get_momentum(ma=ma)
        with pd.HDFStore('momentum_test.h5') as store:
            store.put(f'factor_{ma}', factor)

            summary, total = factor_ic(factor)
            print(total.median())
            store.put(f'ic/{ma}', summary.assign(ma=ma))
            store.put(f'corr/{ma}', total.assign(ma=ma))


print(prices.info())
df = get_momentum(asset_prices=prices)
print(df.info())
print(df.tail())
tune_params()
exit()

with pd.HDFStore('another_test.h5') as store:
    keys = store.keys()
    corr = [k for k in keys if k[1:].startswith('corr')]
    ic = [k for k in keys if k[1:].startswith('ic')]
    corrs = pd.DataFrame()
    for c in corr:
        ms, ml = c.split('_')[1:]
        corrs = pd.concat([corrs, store[c].assign(ms=ms, ml=ml)])

    print(corrs.groupby(['ms', 'ml']).mean())


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
        ab_ = ab.loc[(ab.index.get_level_values('metric') == 'Ann. alpha') & (ab[period] > 0), period].to_frame(
                'alpha')
        best = pd.concat([ic_, ab_.reset_index('metric', drop=True)], axis=1).dropna(how='all').assign(
                period=period)
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


def get_ma_cross(asset_prices: pd.DataFrame, ma_short: int = 10, ma_long: int = 100):
    """Compute dual moving average signal

    :param asset_prices:
    :param ma_short:
    :param ma_long:
    :return:
    """
    mavg = {}
    for ma in [ma_short, ma_long]:
        mavg[ma] = prices.rolling(ma).apply(lambda x: (x[-1] / x[0]) ** (1 / ma) - 1)
        mavg[ma] = mavg[ma].replace([np.inf, -np.inf], np.nan)

    return mavg[ma_short].sub(mavg[ma_long]).dropna(how='all')
