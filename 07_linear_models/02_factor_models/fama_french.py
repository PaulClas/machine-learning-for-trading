#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pprint import pprint
from pandas_datareader.famafrench import get_available_datasets
import pandas as pd
from statsmodels.api import OLS, add_constant
import pandas_datareader.data as web
from pathlib import Path
from linearmodels.asset_pricing import TradedFactorModel, LinearFactorModel, LinearFactorModelGMM

# pprint(get_available_datasets())

data_path = Path('..', '00_data')

ff_factor = 'F-F_Research_Data_5_Factors_2x3_daily'
# ff_data = web.DataReader(ff_factor, 'famafrench')[0]

with pd.HDFStore(data_path / 'risk_factors.h5') as store:
    ff_data = store.get(ff_factor).tz_localize('UTC')
print(ff_data.info())

N_LONGS = 200
N_SHORTS = 0
VOL_SCREEN = 1000
with pd.HDFStore(data_path / 'assets.h5') as store:
    prices = store['quandl/wiki/prices'].sort_index()
    top_vol = prices.loc['2018', 'adj_volume'].unstack().median().nlargest(10)
    returns = prices.adj_close.unstack().filter(top_vol.index).loc['2017':].pct_change().dropna(how='all')
    returns = returns.tz_localize('UTC').dropna(axis=1, thresh=int(len(returns) * .9)).dropna(thresh=int(returns.shape[1] * .9))
    returns = returns.drop(ff_data.columns.intersection(returns.columns), axis=1)
    returns = returns.fillna(returns.median())

# id = f'_long_{N_LONGS}_short_{N_SHORTS}_vol_{VOL_SCREEN}'
# with pd.HDFStore(data_path / 'backtests.h5') as store:
#     pf_returns = store['returns' + id].mul(100).to_frame('Portfolio')

# common_dates = ff_data.index.intersection(pf_returns.index)

common_dates = ff_data.index.intersection(returns.index)
returns = returns.loc[common_dates]
ff_data = ff_data.loc[common_dates]

# data = returns.join(ff_data, how='inner', lsuffix='_ret')


def lin_reg_results(y, X):
    model = OLS(endog=y, exog=X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    return (model.params.to_frame('coef')
            .join(model.conf_int().rename(columns={0: 'lower', 1: 'upper'})))


# factor_exposures = data.groupby(data.index.year).apply(lambda x: lin_reg_results(y=x.Portfolio, X=x.iloc[:, 1:]))
# factor_exposures = factor_exposures.swaplevel(0, 1)
# print(factor_exposures)


# model = OLS(endog=data.Portfolio, exog=data.iloc[:, 1:].assign(alpha=1))
# trained_model = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})
# print(trained_model.summary())
# pprint(dir(trained_model))
# print(trained_model.params)
# print(trained_model.conf_int())

# print(data.info())
# exit()

returns = returns.sub(ff_data.RF, axis=0)
print(returns.info())
print(ff_data.info())

trained_model = TradedFactorModel(returns, ff_data).fit()
# print(trained_model.full_summary)
alphas_traded = trained_model.alphas
betas_traded = trained_model.betas
trained_model = LinearFactorModel(returns, ff_data).fit()
# print(trained_model.full_summary)
alphas_lin = trained_model.alphas
betas_lin = trained_model.betas
print(alphas_traded.equals(alphas_lin))
print(betas_traded.equals(betas_lin))
trained_model = LinearFactorModelGMM(returns, ff_data).fit()
# print(trained_model.full_summary)tra
print(trained_model.alphas)
print(trained_model.betas)
print(trained_model.betas.corr())

# pprint(dir(trained_model))
# print(trained_model.alphas)
# print(trained_model.betas)
# print(trained_model.model)
# print(trained_model.name)
# print(trained_model.params)
# print(trained_model.risk_premia)


"""
comparing cov corrections
             base        hac        hc0        hc1        hc2        hc3
const   -0.831811  -0.804528  -0.792093  -0.790714  -0.790149  -0.788203
Mkt-RF  60.967432  38.429023  40.677390  40.606603  40.485453  40.293231
SMB      9.457036   7.503174   7.908571   7.894808   7.877760   7.846925
HML     -2.458990  -2.024904  -2.140651  -2.136926  -2.132149  -2.123659
RMW     -1.954589  -1.574746  -1.624207  -1.621381  -1.615864  -1.607500
CMA      0.808437   0.742006   0.772242   0.770898   0.768481   0.764703
RF       0.055854   0.066319   0.066540   0.066424   0.066312   0.066085
"""
from sklearn.model_selection import TimeSeriesSplit
