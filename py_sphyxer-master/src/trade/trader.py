
"""

trader class
- performs functions of a treader: receives recommendations, decides trades, evaluates performance
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
from portfolio.portfolio import Portfolio

register_matplotlib_converters()

logger = logging.getLogger(__name__)

HOLDINGS_COLUMN_NAMES =['date', 'tckr_symbol', 'qty', 'fee']


class Trader:

    @log_machine
    def __init__(self, name: str, config: dict, portfolio: Portfolio, df_recomendation: pd.DataFrame = None):

        self.status = 102
        self.exception = te.TkrException()

        self.name = name
        self.score = 0

        self.portfolio = portfolio
        self.df_recommendation = df_recomendation

        self.status = 200

    # def get_portfolio(self):
    #
    #     self.portfolio.get_current_holding()
    #     pass

    def get_recommendation(self):
        pass

    def make_transaction(self):
        pass
