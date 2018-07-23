#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import pandas as pd
from pandas.tseries.offsets import BDay


def get_sp500(start=2010, end=2017):
    with pd.HDFStore('assets.h5') as store:
        sp500 = store['sp500'].Close.loc[str(start):str(end)]
        return sp500.to_frame('sp500')


def get_sp500_ret(start=2010, end=2017, fw=1):
    return get_sp500(start, end).pct_change(periods=fw)


def get_sp500_fw(start=2010, end=2017, periods=(1, 5, 10)):
    sp500 = get_sp500(start, end).resample('D').ffill()
    df = pd.concat([(sp500
                     .pct_change(periods=fw)
                     .shift(-fw)
                     .squeeze()
                     .to_frame(f'{fw}D')) for fw in periods], axis=1)
    return df.dropna()


def get_benchmark(factor_data, start=2013, periods=holding_periods):
    factor_dates = factor_data.index.get_level_values('date').unique()
    last = (factor_dates.max() + BDay(max(periods))).year + 1
    return get_sp500_fw(start=start, end=last, periods=periods).reindex(factor_dates)


def get_quandl_wiki(stocks='all', start=2010, end=2017, na_thres=.8):
    with pd.HDFStore('assets.h5') as store:
        df = store.get('wiki').adj_close.unstack().loc[str(start):str(end)]
        if not isinstance(stocks, str):
            df = df.filter(stocks)
        return df.dropna(thresh=int(len(df) * na_thres), axis=1)


# def get_factor(start=2012, end=2017, ma=24):
#     with pd.HDFStore('momentum_test.h5') as store:
#         return store[f'factor/{ma}'].loc[str(start): str(end)]


def get_factor(start=2012, end=2017, ma_short=5, ma_long=25):
    with pd.HDFStore('momentum_test.h5') as store:
        return store[f'factor_{ma_short}_{ma_long}'].loc[str(start): str(end)]
