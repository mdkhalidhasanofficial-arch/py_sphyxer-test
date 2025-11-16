
import os
import datetime
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

from yahoofinancials import YahooFinancials

from cs_utils import log_machine

from models.arima import ARIMA
import tkr_exception as te
import data_readers as dr

register_matplotlib_converters()

logger = logging.getLogger(__name__)


class Tckr:
    """
    class object to retain values associated to stock market trading symbol
    """

    @log_machine
    def __init__(self, tckr: str, config: dict, force_no_refresh: bool=False):
        """

        :param tckr: (str) stick market trading symbol, e.g., AAPL
        :param config: (dict) required keys:
                        'refresh_data', 'start_date', 'end_date', 'npredict', 'pkl_path',
                        ['output']['reports']
        :param force_no_refresh: (bool), True=do not refresh data source
        """

        self.status = 102

        self.exception = te.TkrException()

        self.symbl = tckr

        self.report_dir = config['output']['reports']
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)

        self.refresh_data = config['refresh_data']

        self.df_price = None
        self.data_quality = {'dq': np.NaN}
        self.df_pred = None

        self.first_date = config['start_date']
        self.last_date = config['end_date']
        self.ntest = config['npredict']
        self.n_rows = None
        self.q10, self.q25, self.q75, self.q90 = None, None, None, None
        self.mean, self.median, self.std, self.rel_stdev = None, None, None, None

        if not os.path.exists(config['pkl_path']):
            os.makedirs(config['pkl_path'])

        # ... columns: ['date', 'high', 'low', 'open', 'close', 'volume', 'adjclose']
        if self.refresh_data and (not force_no_refresh):
            try:
                self.df_price = dr.get_ticker_data(self.symbl, self.first_date, self.last_date)
                self._write_pkl(config['pkl_path'], self.symbl)
                self.status = 200
            except KeyError:
                self.status = 404
        else:
            self.df_price = self._read_pkl(config['pkl_path'])

            last_date = datetime.datetime.strptime(self.last_date, "%Y-%m-%d")
            if self.df_price['date'].min() > last_date or self.df_price['date'].max() < last_date:
                self.df_price = dr.get_ticker_data(self.symbl, self.first_date, self.last_date)
                self._write_pkl(config['pkl_path'], self.symbl)
                self.status = 200

        if self.df_price is not None:
            self.get_tckr_stats()

        # ... identify allowable model types
        self.lx_model_types = {'ARIMA': ARIMA}
        self.lx_model = {}

        # ... instantiate model type objects within Tckr object
        for model_name, model_class in self.lx_model_types.items():
            self.lx_model[model_name] = model_class()

    def _validate_data(self):
        """
        evaluate state of current df_price dataframe
        - if any 'adjclose' values null, sets data_quality[dq] = False
        """

        if self.df_price is not None:
            if self.df_price['adjcose'].isnull().values.any():
                self.data_quality.update({'dq': False})
                logger.warning('Symbol %s - NaNs in adjclose data', self.symbl)



    def get_tckr_stats(self):
        """
        populates basic statistics of df_price[adj_close] values:
        mean, median, std, rel_stdev, q10, q25, q75, q90
        """

        if len(self.df_price):
            self.first_date = self.df_price['date'].min()
            self.last_date = self.df_price['date'].max()
            self.n_rows = len(self.df_price)

            self.q10, self.q25, self.q75, self.q90 = \
                self.df_price.loc[:, 'adjclose'].quantile([0.10, 0.25, 0.75, 0.90]).tolist()

            self.mean = self.df_price.loc[:, 'adjclose'].mean()
            self.median = self.df_price.loc[:, 'adjclose'].median()
            self.std = self.df_price.loc[:, 'adjclose'].std()
            self.rel_stdev = self.std / self.mean * 100

    def add_predictions(self, ls_prediction: list, t_or_f: str = 'pred', concat: bool = False):
        """
        create dataframe from list of predicted values
        records values to .csv file in /reports/ directory

        :param ls_prediction:
        :param t_or_f:
        :param concat: (bool) concatenates (or not) to pre-existing self.df_pred
        :return:
        """

        time_now = datetime.datetime.now()

        # ... add rows to element dataframe for future forecast date
        start_date = self.last_date + datetime.timedelta(days=1)
        end_date = self.last_date + datetime.timedelta(days=self.ntest)
        dt_forecast_dates = pd.date_range(start_date, end_date, freq='d')

        # ... list of % change from last observed
        df_pre_forecast = self.df_price[self.df_price['date'] < start_date]
        last_obs = df_pre_forecast.tail(1).iloc[0]['adjclose']
        ls_delta = [(x - last_obs)/last_obs * 100 for x in ls_prediction]

        # add values to dataframe with predicted values
        df_new = pd.DataFrame({'date': dt_forecast_dates, 'type': self.ntest*[t_or_f],
                               'pred': ls_prediction, 'delta': ls_delta,
                               'time': self.ntest*[time_now]})

        if concat:
            self.df_pred = pd.concat([self.df_pred, df_new])
        else:
            self.df_pred = df_new.copy()

        self.df_pred.reset_index(drop=True, inplace=True)

        fname = self.symbl + '_pred'
        self._write_csv(fname, self.df_pred)

    def get_predictions(self):
        """
        reads predicted values from previously stored csv file
        stores in self.df_pred
        """

        self.df_pred = pd.DataFrame()

        csv_name = self.report_dir + '/ticker/' + self.symbl + '_pred.csv'
        csv_path = Path(csv_name)

        if csv_path.is_file():
            with open(csv_name, 'r') as f_in:
                self.df_pred = pd.read_csv(f_in)

    def _write_csv(self, fname:str, df: pd.DataFrame):
        """
        writes supplied dataframe to report_dir/ticker/fname.csv

        :param fname: (str) filename to write to (without .csv)
        :param df: (pd.DataFrame) dataframe to write
        """

        csv_path = Path(self.report_dir) / 'ticker'
        Path(csv_path).mkdir(parents=True, exist_ok=True)
        csv_name = csv_path / (fname + '.csv')

        with open(csv_name, 'w+') as f_out:
            df.to_csv(f_out, index=False)

    def _write_pkl(self, pkl_path: str, fname: str):
        """
        writes self.df_price dataframe to pkl_path/fname.pkl

        :param pkl_path:
        :param fname: (str) filename to write to (without .pkl)
        """

        pkl_name = Path(pkl_path) / (fname + '.pkl')
        with open(pkl_name, 'wb') as f_out:
            pickle.dump(self.df_price, f_out, pickle.HIGHEST_PROTOCOL)

    def _read_pkl(self, pkl_path):
        """
        reads self.df_price dataframe from pkl_path/self.symbl.pkl

        :param pkl_path: (str)
        :return pd.DataFrame
        :exception FileNotFoundError sets self.status = 404 and logs error
        """

        pkl_name = Path(pkl_path) / (self.symbl + '.pkl')

        try:
            with open(pkl_name, 'rb') as f_in:
                df_price = pickle.load(f_in)
                logger.info('%s pickle file loaded: %d lines' % (self.symbl, len(df_price)))
                self.status = 200
        except FileNotFoundError as fnfe:
            df_price = pd.DataFrame()
            self.status = 404
            logger.error(f'{str(fnfe)}: pickle file not loaded: {pkl_name}')

        return df_price
