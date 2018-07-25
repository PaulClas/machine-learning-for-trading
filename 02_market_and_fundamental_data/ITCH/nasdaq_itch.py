#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import prickle as pk
from prickle import load_hdf5
from urllib.request import urlretrieve
from pprint import pprint
from pathlib import Path
import gzip
import shutil
import h5py
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols

data_path = Path('/drive/data/algo_trading/data/itch')


def may_be_download(url):
    filename = Path(url.split('/')[-1])
    exit()
    unzipped = filename.stem + '.bin'
    if not filename.exists():
        urlretrieve(url, filename)
    if not Path(unzipped).exists():
        with gzip.open(str(filename), 'rb') as f_in:
            with open(unzipped, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    return unzipped


# url = 'ftp://emi.nasdaq.com/ITCH/GIS/S022817-v50.txt.gz'
# url = 'ftp://emi.nasdaq.com/ITCH/01302017.NASDAQ_ITCH50.gz'
# file = may_be_download(url)

def unpack_itch(file_name, names, date, nlevels=10, ver=5.0):
    pk.unpack(fin=str(data_path / file_name),
              ver=ver,
              date=date,
              fout=data_path / 'itch3.hdf5',
              nlevels=nlevels,
              names=names,
              method='hdf5')


symbols = get_nasdaq_symbols()
# names = symbols.index.tolist()
names = ['AAPL']
file_name = '03292018.NASDAQ_ITCH50'
date = '03292018'

unpack_itch(file_name=file_name, date=date, names=names)
exit()

types = {0: 'add',
         1: 'add_mpid',
         2: 'cancel',
         3: 'delete',
         4: 'exec',
         5: 'exec_p',
         6: 'replace'}

name = 'AAPL'
hdf = data_path / 'itch.hdf5'
messages = load_hdf5(hdf, grp='messages', name=name)
print(messages.info())
print(messages.type.map(types).value_counts())
exit()
price, volume = load_hdf5(hdf, grp='books', name=name)
print(price.info())
print(volume.info())
