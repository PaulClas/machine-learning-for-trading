#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

"""
%%zipline --start 2010-1-1 --end 2018-1-1 -o results/results_5_50_95_0.pickle
"""

import pandas as pd

from zipline.finance import commission, slippage
from zipline.api import get_datetime, set_benchmark

MA_SHORT = 10
MA_LONG = 50
BOTTOM_Q = .025
TOP_Q = .95
MAX_SHORT = -0.25
MAX_LONG = 0.25

with pd.HDFStore('assets.h5') as store:
    STOCKS = store.get('quandl/wiki/stocks/sp500').symbol


def initialize(context):
    context.stocks = STOCKS
    context.sids = [context.symbol(symbol) for symbol in context.stocks]

    context.i = 0
    context.ma_short = MA_SHORT
    context.ma_long = MA_LONG
    context.top_q = TOP_Q
    context.bottom_q = BOTTOM_Q
    context.max_s = MAX_SHORT
    context.max_l = MAX_LONG
    context.years = []

    context.set_commission(commission.PerShare(cost=0))
    context.set_slippage(slippage.FixedSlippage(spread=0))


def handle_data(context, data):
    lt = context.ma_long
    context.i += 1
    if context.i < lt:
        return

    now = get_datetime()
    ma = {'st': context.ma_short, 'lt': lt}
    stocks = [s for s in context.sids if s.start_date <= now <= s.end_date]

    hist = {m: data.history(stocks, 'price', bar_count=w, frequency='1d') for m, w in ma.items()}
    mavg = {m: hist[m].add(1).fillna(1).prod().sub(1).pow(1 / hist[m].count()) for m, w in ma.items()}
    std = hist['lt'].std()

    dev = mavg['st'].sub(mavg['lt'])
    q = dev.quantile(q=context.top_q)

    long = (dev >= max(q[context.top_q], 0)).astype(int)
    long = long.div(std).clip(lower=0, upper=context.max_l)
    if long.sum() > 1:
        long = long.div(long.sum())

    rebalance_portfolio(context, long.fillna(0).to_dict())


def rebalance_portfolio(context, positions):
    for sid, position in positions.items():
        context.order_target_percent(sid, position)
