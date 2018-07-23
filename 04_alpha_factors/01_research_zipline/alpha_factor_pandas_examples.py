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
    """Load quandl wiki sample and adjust to NYSE trading calendar"""
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
    """

    :param asset_prices:
    :param relative_lookback:
    :param absolute_lookback:
    :param quantiles:
    :return:
    """
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


short_term = 10
long_term = 50
for which in ['sp500', 'all']:
    print(which)
    with pd.HDFStore('momentum_factor.h5') as store:
        factor_data = store[f'mean_reversal/{short_term}/{long_term}/{which}/factor_data'].loc[idx['2002':'2017', :], :]
    qmin, qmax = factor_data.factor_quantile.min(), factor_data.factor_quantile.max()

    input_data = create_pyfolio_input(factor_data,  # generated using alphalens
                                      period='1D',  # holding period
                                      capital=100000,  # starting capital
                                      long_short=False,  # market-neutral pf if True
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