from pathlib import Path
import pandas as pd
from dateutil.relativedelta import relativedelta

idx = pd.IndexSlice
with pd.HDFStore('../data/assets.h5') as store:
    stocks = store['quandl/wiki/stocks']
    prices = store['quandl/wiki/prices'].adj_close

sec = pd.read_csv('report_index.csv').rename(columns=str.lower)
sec.date_filed = pd.to_datetime(sec.date_filed)
first = sec.date_filed.min() + relativedelta(months=-1)
last = sec.date_filed.max() + relativedelta(months=1)
prices = (prices
          .loc[idx[first:last, :]]
          .unstack().resample('D')
          .ffill()
          .dropna(how='all', axis=1)
          .filter(sec.ticker.unique()))
sec = sec.loc[sec.ticker.isin(prices.columns), ['ticker', 'date_filed']]

price_data = []
for ticker, date in sec.values.tolist():
    target = date + relativedelta(months=1)
    s = prices.loc[date: target, ticker]
    price_data.append(s.iloc[-1] / s.iloc[0] - 1)

df = pd.DataFrame(price_data,
                  columns=['returns'],
                  index=sec.index)

print(df.returns.describe())
sec['returns'] = price_data
print(sec.info())
sec.dropna().to_csv('sec_returns.csv', index=False)
