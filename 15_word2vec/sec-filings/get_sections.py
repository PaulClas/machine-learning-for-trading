#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import numpy as np
import pandas as pd
from pprint import pprint
import string

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)


with pd.HDFStore('../../data/assets.h5') as store:
    stocks = store['quandl/wiki/stocks']

filing_path = Path('reports')

filings = pd.read_csv('report_index.csv').rename(columns=str.lower)
filings = (filings[filings.ticker.isin(stocks.symbol)].index + 1).tolist()
items_path = Path('report_items')
if not items_path.exists():
    items_path.mkdir(exist_ok=True)

i = 1
for filing in filing_path.glob('*.txt'):
    if i % 500 == 0:
        print(i, end=' ', flush=True)
    filing_id = int(filing.stem)
    if filing_id not in filings:
        continue
    items = {}
    for section in filing.read_text().lower().split('°'):
        if section.startswith('item '):
            try:
                item = section.split()[1].replace('.', '')
            except IndexError:
                print(section)
                continue
            text = ' '.join([t for t in section.split()[2:]])
            if items.get(item) is None or len(items.get(item)) < len(text):
                items[item] = text

    txt = pd.Series(items).reset_index()
    txt.columns = ['item', 'text']
    txt.to_csv(items_path / (filing.stem + '.csv'), index=False)
    i += 1
