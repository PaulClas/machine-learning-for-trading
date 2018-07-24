#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import pandas as pd
import numpy as np
import warnings
from alphalens.performance import factor_information_coefficient, mean_information_coefficient, create_pyfolio_input, \
    mean_return_by_quantile
from alphalens.utils import get_clean_factor_and_forward_returns
from pathlib import Path

np.random.seed(42)

warnings.filterwarnings('ignore')
pd.set_option('display.expand_frame_repr', False)

idx = pd.IndexSlice
TRADING_DAYS = 21
holding_periods = [m * TRADING_DAYS for m in [3, 6, 9, 12]]

FACTOR_STORE = Path('..', '01_alpha_factors', 'momentum_factor.h5')
PYFOLIO_STORE = 'risk_metrics.h5'


def make_pyfolio(which='sp500'):
    print(which)
    for lookback in [12, 18, 24, 36]:
        print(f'\n\tLookback: {lookback} | ', end=' ')
        factor_key = f'momentum_overlay/factor_data/{lookback}/{which}'
        factor_data = pd.read_hdf(FACTOR_STORE, factor_key).loc[idx['2013': '2017', :], :]

        for holding_period in holding_periods:
            print(f'{holding_period}', end=' ', flush=True)
            for long_short in [True, False]:
                for top in [True, False]:
                    quantiles = [1, 5] if top else None
                    returns, positions, benchmark = create_pyfolio_input(factor_data,
                                                                         period=f'{holding_period}D',
                                                                         capital=1e6,
                                                                         long_short=long_short,
                                                                         group_neutral=False,
                                                                         equal_weight=False,
                                                                         quantiles=quantiles,
                                                                         groups=None,
                                                                         benchmark_period=f'{holding_period}D')

                    pf_key = f'{which}/{lookback}/{holding_period}/{int(long_short)}/{int(top)}/'
                    with pd.HDFStore(PYFOLIO_STORE) as store:
                        store.put(pf_key + 'returns', returns)
                        store.put(pf_key + 'positions', positions)
                        store.put(pf_key + 'benchmark', benchmark)


make_pyfolio('all')
