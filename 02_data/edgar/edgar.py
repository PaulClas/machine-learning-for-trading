#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import pandas as pd

data_dir = Path('/drive/data/algo_trading/data/edgar')
fs = '2018q1'

sub = pd.read_csv(data_dir / fs / 'sub.txt', sep='\t')
print(sub.info())
print(sub.head())
print(sub[sub.name.str.lower().str.contains('alphabet')].T.squeeze())
adsh = '0001652044-18-000007'
# print(sub.form.value_counts())
# exit()
pre = pd.read_csv(data_dir / fs / 'pre.txt', sep='\t')
print(pre.info())
print(pre.head())
data = pre[pre.adsh == adsh]
tag = pd.read_csv(data_dir / fs / 'tag.txt', sep='\t')
print(tag.info())
# print(tag.head())
data.merge(tag, how='left').to_csv('tags.csv')
num = pd.read_csv(data_dir / fs / 'num.txt', sep='\t')
num = num[num.adsh == adsh]
print(num.info())
print(num.head())
num.to_csv('num.csv')

