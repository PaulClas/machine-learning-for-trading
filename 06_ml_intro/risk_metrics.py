#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

"""
Common financial risk and performance metrics. 
Used by zipline and pyfolio. 
http://quantopian.github.io/empyrical
"""

import pandas as pd
import pandas_datareader.data as web
from pprint import pprint
from pandas_datareader.famafrench import get_available_datasets
from pathlib import Path

data_dir = Path('..', 'data')


def get_ff():
    datasets = get_available_datasets()

    of_interest = ['F-F_Research_Data_5_Factors_2x3_daily',
                   '6_Portfolios_2x3_daily',
                   '25_Portfolios_5x5_Daily',
                   '100_Portfolios_10x10_Daily']

    for pf in of_interest:
        df = web.DataReader(pf, 'famafrench')
        print(df['DESCR'])
        print(df[0].columns.tolist())

