#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from datetime import date, datetime
import pandas as pd
import requests
from pathlib import Path
from collections import namedtuple, Counter
from bs4 import BeautifulSoup

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 10)
pd.set_option('display.precision', 2)

data_path = Path('/drive/data/algo_trading/data/edgar/filings')


def download_index(first_year=1993, last_year=2018, last_quarter=2, index='master'):
    past_years = range(first_year, last_year)
    filing_periods = [(y, q) for y in past_years for q in range(1, 5)]
    filing_periods.extend([(this_year, q) for q in range(1, last_quarter + 1)])

    record = namedtuple('filing', field_names=['cik', 'company', 'form', 'date', 'url'])

    filings = []
    index_url = 'https://www.sec.gov/Archives/edgar/full-index/{}/QTR{}/{}.idx'
    for p, (yr, qtr) in enumerate(filing_periods):
        f = requests.get(index_url.format(yr, qtr, index), timeout=30).text
        lines = f.splitlines()[11:]  # start of file body
        print(p, yr, qtr, '{:,}'.format(len(lines)))
        filings.extend([record._make(l.split('|')) for l in lines])

    filings = pd.DataFrame(filings)
    filings.date = pd.to_datetime(filings.date)
    filings.to_parquet(data_path / 'filings' / 'xbrl' / '{}.parquet'.format(yr))


today = pd.Timestamp(date.today())
this_year = today.year
this_quarter = today.quarter
# download_index(last_year=this_year, last_quarter=this_quarter)


def get_apple_filings():
    data = pd.read_parquet(data_path / 'index' / 'filing_index.parquet')
    data = data[(data.company.str.lower().str.contains('apple inc')) & (data.form == '10-K')]
    data['year'] = data.date.dt.year
    data['quarter'] = data.date.dt.quarter
    print(data.groupby(['year', 'quarter', 'form']).size())

archive_url = 'https://www.sec.gov/Archives/'
# file_url = 'edgar/data/320193/0001193125-13-416534.txt'
file_url = 'edgar/data/320193/0001628280-16-020309.txt'
# exit()
# for url in data.sort_values('year', ascending=False).url:
#     f = requests.get(archive_url + url).text

f = requests.get(archive_url + file_url).text
soup = BeautifulSoup(f, 'lxml')
for t, table in enumerate(soup.find_all('table')):
    try:
        print(pd.read_html(str(table))[0].dropna(how='all').dropna(axis=1, how='all').head())
    except:
        print('error')

    if t > 50:
        break
exit()
tags = Counter([t.name for t in soup.find_all()])
tags = pd.Series(tags)
print(tags.sort_values(ascending=False).head(15))
# tags.sort_values(ascending=False).to_frame('count').to_csv('tags.csv')
print(tags.describe())
