#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import pandas as pd
import numpy as np

df = pd.read_html('https://www.barrons.com/articles/penta-top-100-hedge-funds-1497665963?tesla=y&mod=article_inline', header=0)[0].iloc[:, 2:]
df.columns =['fund', 'fund_assets', 'strategy', '3yr return', '2016 return', 'firm', 'firm_assets']
for col in ['fund_assets', 'firm_assets']:
     df[col] = df[col].str.replace('$', '').str.replace(',', '').str.split(expand=True).iloc[:, 0].apply(pd.to_numeric)
print(df.info())
df.to_csv('hedge_funds.csv', index=False)


exit()

d = {'home': ['A', 'B', 'B', 'A', 'B', 'A', 'A'], 'away': ['B', 'A', 'A', 'B', 'A', 'B', 'B'],
     'aw'  : [1, 0, 0, 0, 1, 0, np.nan], 'hw': [0, 1, 0, 1, 0, 1, np.nan]}

df = pd.DataFrame(d, columns=['home', 'away', 'hw', 'aw'])
df.index = range(1, len(df) + 1)

print(df)
# df['homewin_at_home'] = df.groupby('home')['hw'].apply(lambda x: pd.expanding_mean(x).shift())
df['homewin_at_home'] = df.groupby('home')['hw'].apply(lambda x: x.expanding().mean().shift())

print(df)
