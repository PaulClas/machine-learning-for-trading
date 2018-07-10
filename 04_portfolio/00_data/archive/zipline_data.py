#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import os
from pandas import Timestamp
from pprint import pprint
import pandas as pd
import toolz
from datetime import datetime
from zipline.data.bundles import load
from zipline.pipeline import Pipeline
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.data import USEquityPricing

now = Timestamp.utcnow()
# bundle = load('quantopian-quandl', os.environ, now)
# assets = bundle.asset_finder.retrieve_all(bundle.asset_finder.sids)
# print(assets[:5])
# test_asset = assets[0]
# print(pd.Series(test_asset.to_dict()))
# pprint([d for d in dir(assets[0]) if not d.startswith('_')])

# symbols = [asset.symbol for asset in assets]


# print(len(symbols))
# print(type(symbols[0]))


@toolz.memoize
def _pipeline_engine_and_calendar_for_bundle(bundle):
    """Create a pipeline engine for the given bundle.

    Parameters
    ----------
    bundle : str
        The name of the bundle to create a pipeline engine for.

    Returns
    -------
    engine : zipline.pipleine.engine.SimplePipelineEngine
        The pipeline engine which can run pipelines against the bundle.
    calendar : zipline.utils.calendars.TradingCalendar
        The trading calendar for the bundle.
    """
    bundle_data = load(bundle)
    pipeline_loader = USEquityPricingLoader(
            bundle_data.equity_daily_bar_reader,
            bundle_data.adjustment_reader
    )

    def choose_loader(column):
        if column in USEquityPricing.columns:
            return pipeline_loader
        raise ValueError(
                'No PipelineLoader registered for column %s.' % column
        )

    calendar = bundle_data.equity_daily_bar_reader.trading_calendar
    return (
        SimplePipelineEngine(
                choose_loader,
                calendar.all_sessions,
                bundle_data.asset_finder,
        ),
        calendar,
    )


def run_pipeline_against_bundle(pipeline, start_date, end_date, bundle):
    """Run a pipeline against the data in a bundle.

    Parameters
    ----------
    pipeline : zipline.pipeline.Pipeline
        The pipeline to run.
    start_date : pd.Timestamp
        The start date of the pipeline.
    end_date : pd.Timestamp
        The end date of the pipeline.
    bundle : str
        The name of the bundle to run the pipeline against.

    Returns
    -------
    result : pd.DataFrame
        The result of the pipeline.
    """
    engine, calendar = _pipeline_engine_and_calendar_for_bundle(bundle)

    start_date = pd.Timestamp(start_date, tz='utc')
    if not calendar.is_session(start_date):
        # this is not a trading session, advance to the next session
        start_date = calendar.minute_to_session_label(
                start_date,
                direction='next',
        )

    end_date = pd.Timestamp(end_date, tz='utc')
    if not calendar.is_session(end_date):
        # this is not a trading session, advance to the previous session
        end_date = calendar.minute_to_session_label(
                end_date,
                direction='previous',
        )

    return engine.run_pipeline(pipeline, start_date, end_date)


start = pd.to_datetime('2002')
end = pd.to_datetime('2018')
pipeline = Pipeline({'close': USEquityPricing.close.latest})
data = run_pipeline_against_bundle(pipeline, start_date=start, end_date=end, bundle='quantopian-quandl').unstack()['close']
data.columns = [c.symbol for c in data.columns]
data = data.resample('B').last()

with pd.HDFStore('assets2.h5') as store:
    store.put('quandl', data)
