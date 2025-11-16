
"""

"""
import os
import sys
import datetime
import logging
import platform
from pathlib import Path

import numpy as np
import pandas as pd

import constants as c
import cs_utils as cu
from cs_utils import Timer, replace_text_line
from viz.plotter_plotly import make_time_series_plot
from ticker.tkr import Tckr

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

logger = logging.getLogger(__name__)

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... orchestrate model sequence
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def orchestrate_modeler(config) -> int:

    status = 102

    fpath = Path(config['output']['plot_dir'])
    Path(fpath).mkdir(parents=True, exist_ok=True)

    n_train = config['ntrain']
    n_test = config['npredict']

    # ... build forecasts for each ticker symbol
    for this_tckr in config['ls_tckr']:

        # ... instantiate symbol and price history and attaches valid model types objects
        logger.info('collect history of %s', this_tckr)
        this_eqty = Tckr(this_tckr, config)

        if this_eqty.status > 399:
            logger.error(f'Status message: {this_eqty.status} returned during initialization of {this_tckr}')
            continue

        # ... assign tkr price history to local df name
        df_tckr = this_eqty.df_price.copy()

        logger.info('Begin model: %s', this_tckr)

        # ... create model
        logger.info(f'create model {this_tckr}')
        this_model = this_eqty.lx_model['ARIMA']

        # ... prepare data to model requirements (need to expand this if multuple model types imlemented)
        this_model.prepare_data(this_eqty)

        # ... tune model
        logger.info(f'tune model {this_tckr}')
        this_model.tune(this_eqty.symbl, 'adjclose', n_train, n_test)

        # ... fit model
        logger.info(f'fit model {this_tckr}')
        this_model.fit('adjclose')

        # ... forecast
        logger.info(f'forecast  {this_tckr}')
        ls_predictions = this_model.predict(n_test)

        # ... add model predictions to Tckr dataframe
        this_eqty.add_predictions(ls_predictions)

        # ... make time series plot
        logger.info(f'time-series plot {this_tckr}')
        df_plot = pd.concat([this_eqty.df_price, this_eqty.df_pred], axis=0)
        fig, cnfg = make_time_series_plot(df_plot, this_eqty.symbl, ['adjclose', 'pred'])
        # fig, cnfg = make_time_series_plot(df_plot, this_eqty.symbl, ['close', 'pred'])

        html_path = fpath / str(this_eqty.symbl + '_pred.html')
        fig.write_html(str(html_path), include_plotlyjs=False)
        replace_text_line(html_path, '<div>', c.PLOTLY_JAVA)

    # ... rank % change in forecasts at n-day horizon
    logger.info('rank forecasts % change')

    df_delta = pd.DataFrame()

    for this_tckr in config['ls_tckr']:
        this_eqty = Tckr(this_tckr, config, force_no_refresh=True)
        if this_eqty.status > 399:
            logger.error(f'Status message: {this_eqty.status} returned during initialization of {this_tckr}')
            continue

        this_eqty.get_predictions()
        df_tckr = this_eqty.df_pred

        if len(df_tckr):
            last_delta = df_tckr.tail(1).iloc[0]['delta']
            this_row = pd.DataFrame(data={'symbl': this_tckr, 'delta': last_delta}, index=[0])
            df_delta = pd.concat([df_delta, this_row])
        else:
            print('*** %s - no predictions' % this_tckr)

    # ... compare each ticker pct_delta to index ticker pct_delta
    if len(df_delta):
        indx_tckr = config['indx_tckr']
        indx_delta = df_delta.loc[df_delta['symbl'] == indx_tckr, 'delta'].values[0]

        df_delta['delta_from_index'] = (df_delta['delta'] - indx_delta) / abs(indx_delta) * 100

        df_delta.sort_values(by='delta_from_index', ascending=False, inplace=True)
        print(df_delta)

        report_dir = config['output']['reports']
        f_delta = Path(report_dir) / ('ticker_ftw_' + (str(config['end_date'])) + '.csv')
        df_delta.to_csv(f_delta, index=False)
    else:
        print('*** df_delta - no values')

    status = 200

    return status
