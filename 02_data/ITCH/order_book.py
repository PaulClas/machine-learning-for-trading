#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import pandas as pd
from pathlib import Path
import numpy as np
from collections import Counter
from time import time
from datetime import timedelta

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 10)
pd.set_option('display.precision', 2)

import pandas_datareader.data as web
import datetime
from pprint import pprint
book = web.get_iex_book('AAPL')
orders = pd.concat([pd.DataFrame(book[side]).assign(side=side) for side in ['bids', 'asks']])
print(orders.sort_values('timestamp').head())
exit()
stock = 'AAPL'
date = '20180329'
order_dict = {-1: 'sell', 1: 'buy'}

data_path = Path('/drive/data/algo_trading/data/itch')
file_name = '03292018.NASDAQ_ITCH50'
itch_store = str(data_path / 'itch.h5')
stock_store = str(data_path / '{}.h5'.format(stock.lower()))


# messages = pd.read_csv('message_labels.csv', index_col=0, squeeze=True).to_dict()


def get_messages(date, stock=stock):
    with pd.HDFStore(itch_store) as store:
        stock_locate = store.select('R', where='stock = stock').stock_locate.iloc[0]
        target = 'stock_locate = stock_locate'

        data = {}
        messages = ['A', 'F', 'E', 'C', 'X', 'D', 'U', 'P', 'Q']
        for m in messages:
            data[m] = store.select(m, where=target).drop('stock_locate', axis=1).assign(type=m)

    order_cols = ['order_reference_number', 'buy_sell_indicator', 'shares', 'price']
    orders = pd.concat([data['A'], data['F']], sort=False, ignore_index=True).loc[:, order_cols]

    for m in messages[2: -3]:
        data[m] = data[m].merge(orders, how='left')

    data['U'] = data['U'].merge(orders, how='left',
                                right_on='order_reference_number',
                                left_on='original_order_reference_number',
                                suffixes=['', '_replaced'])

    data['Q'].rename(columns={'cross_price': 'price'}, inplace=True)
    data['X']['shares'] = data['X']['cancelled_shares']
    data['X'] = data['X'].dropna(subset=['price'])

    data = pd.concat([data[m] for m in messages], ignore_index=True, sort=False)
    data['date'] = pd.to_datetime(date)
    data.timestamp = data['date'].add(data.timestamp)
    data = data[data.printable != 0]

    drop_cols = ['tracking_number', 'order_reference_number', 'original_order_reference_number',
                 'cross_type', 'new_order_reference_number', 'attribution', 'match_number',
                 'printable', 'date', 'cancelled_shares']
    return data.drop(drop_cols, axis=1).sort_values('timestamp').reset_index(drop=True)


# messages = get_messages(date=date)
with pd.HDFStore(stock_store) as store:
    # print(store.info())
    key = '{}/messages'.format(stock)
    # store.put(key, messages)
    messages = store[key]
print(messages.info())


def get_trades(m):
    """Combine C, E, P and Q messages into trading records"""
    trade_dict = {'executed_shares': 'shares', 'execution_price': 'price'}
    cols = ['timestamp', 'executed_shares']
    trades = pd.concat([m.loc[m.type == 'E', cols + ['price']].rename(columns=trade_dict),
                        m.loc[m.type == 'C', cols + ['execution_price']].rename(columns=trade_dict),
                        m.loc[m.type == 'P', ['timestamp', 'price', 'shares']],
                        m.loc[m.type == 'Q', ['timestamp', 'price', 'shares']].assign(cross=1),
                        ], sort=False).dropna(subset=['price']).fillna(0)
    return trades.set_index('timestamp').sort_index().astype(int)


# trades = get_trades(messages)
# print(trades.info())
# with pd.HDFStore(stock_store) as store:
#     store.put('{}/trades'.format(stock), trades)


def add_orders(orders, buysell, nlevels):
    new_order = []
    items = sorted(orders.copy().items())
    if buysell == 1:
        items = reversed(items)
    for i, (p, s) in enumerate(items, 1):
        new_order.append((p, s))
        if i == nlevels:
            break
    return orders, new_order


def save_orders(orders, append=False):
    cols = ['price', 'shares']
    for buysell, book in orders.items():
        df = pd.concat([pd.DataFrame(data=data,
                                     columns=cols).assign(timestamp=t) for t, data in book.items()])
        print(df.info())
        key = '{}/{}'.format(stock, order_dict[buysell])
        with pd.HDFStore(stock_store) as store:
            if append:
                store.append(key, df.set_index('timestamp'), format='t')
            else:
                store.put(key, df.set_index('timestamp'))


order_book = {-1: {}, 1: {}}
current_orders = {-1: Counter(), 1: Counter()}
message_counter = Counter()
nlevels = 100

start = time()
for message in messages.itertuples():
    i = message[0]
    if i % 1e5 == 0 and i > 0:
        print('{:,.0f}\t{}'.format(i, timedelta(seconds=time() - start)))
        # save_orders(order_book, append=True)
        # order_book = {-1: {}, 1: {}}
        start = time()
    if np.isnan(message.buy_sell_indicator):
        continue
    message_counter.update(message.type)

    buysell = message.buy_sell_indicator
    price, shares = None, None

    if message.type in ['A', 'F', 'U']:
        price = int(message.price)
        shares = int(message.shares)

        current_orders[buysell].update({price: shares})
        current_orders[buysell], new_order = add_orders(current_orders[buysell], buysell, nlevels)
        order_book[buysell][message.timestamp] = new_order

    if message.type in ['E', 'C', 'X', 'D', 'U']:
        if message.type == 'U':
            if not np.isnan(message.shares_replaced):
                price = int(message.price_replaced)
                shares = -int(message.shares_replaced)
        else:
            if not np.isnan(message.price):
                price = int(message.price)
                shares = -int(message.shares)

        if price is not None:
            current_orders[buysell].update({price: shares})
            if current_orders[buysell][price] <= 0:
                current_orders[buysell].pop(price)
            current_orders[buysell], new_order = add_orders(current_orders[buysell], buysell, nlevels)
            order_book[buysell][message.timestamp] = new_order

message_counter = pd.Series(message_counter)
print(message_counter)
save_orders(order_book)
with pd.HDFStore(stock_store) as store:
    store.put('{}/summary'.format(stock), message_counter)

# s = sell.groupby(level='timestamp', group_keys=False).apply(lambda x: x.nsmallest(n=10, columns='price'))
# print(s.set_index('price', append=True).unstack().loc[:, 'shares'].info())
# stacked = pd.DataFrame()
# for h, data in df.groupby(df.index.hour):
#     stacked = pd.concat([stacked, data.stack().to_frame('shares')])
# stacked.index.names = ['timestamp', 'price']
# store.put('{}/{}'.format(stock, v), stacked.reset_index('price'))

