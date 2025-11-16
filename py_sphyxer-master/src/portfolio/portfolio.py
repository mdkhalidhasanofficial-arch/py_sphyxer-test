
"""
portfolio class
- df_holdings =  data frame of historical transactions: [date, ticker_symbol, qty, fee]
- optionally, fpath to a .pkl or .csv file can be supplied in lieu of dataframe
"""

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
import constants as c
import tkr_exception as te
from tkr import Tckr

register_matplotlib_converters()

logger = logging.getLogger(__name__)

HOLDINGS_COLUMN_NAMES = ['date', 'tckr_symbol', 'qty', 'fee']


class Portfolio:

    @log_machine
    def __init__(self, config, df: pd.DataFrame = pd.DataFrame(), fpath_holdings=None):
        """
        initialize Portfolio object
        - if df is provided - assigned to portfolio df_holdings
        - if fpath_holdings is supplied AND df is empty or not provided - file read to define holdings
        - if df has no length, AND fpath_holdings is None, then empty data frame is initialized
        :param config:
        :param df:
        :param fpath_holdings:
        """

        self.status = 102
        self.exception = te.TkrException()

        self.df_current = None
        self.value = 0

        # ... initiate holdings definition, depending on input arguments
        if len(df):
            self.df_transactions = df
        elif fpath_holdings is not None:
            ext = fpath_holdings[len(fpath_holdings) - 3:]

            # ... read csv file
            if ext == 'csv':
                self.df_transactions = pd.read_csv(fpath_holdings)
            # ... or read pkl file
            elif ext == 'pkl':
                self.df_transactions = self._read_pkl(fpath_holdings)
            else:
                self.status = 400
                logger.error(f'filepath supplied not .csv or .pkl: {fpath_holdings}')
        else:
            self.df_transactions = pd.DataFrame(columns=HOLDINGS_COLUMN_NAMES)

        # ... verify df_holdings has correct columns
        ls_cols = self.df_transactions.columns.tolist()
        if not set(HOLDINGS_COLUMN_NAMES).issubset(set(ls_cols)):
            logger.error(f'Column names during portfolio initiation are incomplete.'
                         f'supplied: {ls_cols} vs. required: {HOLDINGS_COLUMN_NAMES}')

        # ... assemble dict of key=ticker symbol, value = ticker historical price records
        if len(self.df_transactions):
            # ... assign historic data
            self.ls_symbol = self.df_transactions['tckr_symbol'].unique().tolist()
            self.lx_tckr = {}

            try:
                for t in self.ls_symbol:
                    self.lx_tckr.update({t: Tckr(t, config)})
            except Exception as e:
                self.status = 400
                logger.error(f'error during attempt to join Tckr objects during Portfolio init: {self.status} | {str(e)}')

            # ... updates values and total value to most recent closing prices
            self.get_current_holding()

        self.status = 200

    def record_transaction(self, config, tdate, symbl, qty, fee) -> None:
        """
        record a transaction in the portfolio object holding dataframe

        this does not record the price at time of transaction, valuations can change (adjclose)
        subsequent to a transaction, so the value is updated at using latest available data for adjclose

        :param config (dict)
        :param tdate: datetime object
        :param symbl: valid ticker symbol
        :param qty: (float) positive for buy, negative value to sell
        :param fee:
        :return: None
        """

        # ... concat new info as next row in df_transactions data frame
        new_row = pd.DataFrame({'date': tdate,
                                'tckr_symbol': symbl,
                                'qty': qty,
                                'fee': fee
                                }, index=[0])

        self.df_transactions = pd.concat([self.df_transactions, new_row])

        # ... check if symbl Tckr already added to Portfolio ... or add as necessary
        if symbl not in self.ls_symbol:

            self.lx_tckr.update({symbl: Tckr(symbl, config, force_no_refresh=True)})

            # ... add symbol to list of symbols, and drop duplicates
            self.ls_symbol.append(symbl)
            self.ls_symbol = list(set(self.ls_symbol))

        # ... update current holdings table and total value
        self.get_current_holding()

        return None

    def get_current_holding(self, status_date: datetime = datetime.datetime.now()) -> None:
        """
        iterate over each symbol in portfolio to determine current composition and values

        :param status_date (datetime obj) requested date for portfolio composition
        :return: None
        """

        df_current = pd.DataFrame()
        current_value = 0

        # ... filter portfolio to all transactions prior to requested status date
        df_holding = self.df_transactions[self.df_transactions['date'] <= status_date]

        for symbl in self.ls_symbol:
            symbl_qty = df_holding[df_holding['tckr_symbol'] == symbl]['qty'].sum()

            this_tkr = self.lx_tckr[symbl]
            df_tkr_price = this_tkr.df_price[this_tkr.df_price['date'] <= status_date]
            current_price = df_tkr_price['adjclose'].tail(1).values[0]

            tckr_value = current_price * symbl_qty

            df_row = pd.DataFrame({'date': status_date,
                                   'symbol': symbl,
                                   'qty': symbl_qty,
                                   'adjclose': current_price,
                                   'value': tckr_value}, index=[0])

            df_current = pd.concat([df_current, df_row])

        self.value = df_current['value'].sum()
        self.df_current = df_current

        return None

    def _read_pkl(self, fpath: str) -> pd.DataFrame:
        """
        read previously stored .pkl file
        :param fpath: (str) path to .pkl file to read
        :return: (pd.DataFrame) dataframe from .pkl file

        self.status = 200 for success
        self.status set to 404 in the case of FileNotFoundError and returns empty dataframe
        """

        pkl_name = Path(fpath)

        try:
            with open(pkl_name, 'rb') as f_in:
                df = pickle.load(f_in)
                logger.info(f'{pkl_name} pickle file loaded: {len(df)} lines')
                self.status = 200
        except FileNotFoundError as fnfe:
            df = pd.DataFrame()
            self.status = 404
            logger.error(f'{str(fnfe)}: pickle file not loaded: {pkl_name}')

        return df
