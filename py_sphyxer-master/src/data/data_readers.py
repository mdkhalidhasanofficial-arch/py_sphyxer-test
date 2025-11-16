import logging
import pandas as pd
from pathlib import Path

from yahoofinancials import YahooFinancials

from cs_utils import log_machine


@log_machine
def get_ticker_data(symbl: str, start_date: str, end_date: str) -> pd.DataFrame:
    """

    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna
    aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
    Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

    :param symbl: ticker symbol
    :param start_date: date in str format
    :param end_date:
    :return:
    """


    logger = logging.getLogger(__name__)
    logger.info(f'_get_ticker_data() {symbl} | {start_date} : {end_date}')

    if not start_date < end_date:
        raise ValueError(f'end date must be > start date in {str(__name__)}')

    df_tckr = pd.DataFrame()
    lx_yf_tckr = {}

    try:
        yahoo_financials = YahooFinancials(symbl)
        if not isinstance(start_date, str):
            start_date = start_date.strftime("%Y-%m-%d")
            end_date = end_date.strftime("%Y-%m-%d")
        lx_yf_tckr = yahoo_financials.get_historical_price_data(start_date=start_date,
                                                                end_date=end_date,
                                                                time_interval='daily')
    except TypeError as te:
        logger.error('TypeError in yahoo get price data: %s | %s' % (symbl, str(te)))

    try:
        df_tckr = pd.DataFrame(lx_yf_tckr[symbl]['prices'])
        df_tckr = df_tckr.drop('date', axis=1)
        df_tckr.rename({'formatted_date': 'date'}, axis=1, inplace=True)
        df_tckr['date'] = pd.to_datetime(df_tckr['date'], format="%Y-%m-%d")

        # ... re-order columns
        df_tckr = df_tckr[['date'] + [col for col in df_tckr.columns if col != 'date']]

    except KeyError as ke:
        logger.error(f'{str(ke)}: *** prices not found *** for {symbl}')
        raise KeyError(f'*** prices not found *** for {symbl}')

    return df_tckr


def get_model_pdq(f_path: Path) -> pd.DataFrame:
    """
    reads csv file from f_path, returns data frame

    :param f_path:
    :return: pd.DataFrame
    """

    logger = logging.getLogger(__name__)

    try:
        df = pd.read_csv(f_path)

    except FileNotFoundError as fnfe:
        df = pd.DataFrame()
        logger.error('File not found - %s' % str(fnfe))

    return df


def clean_column_names(ls_col_names: list, chars_to_remove: list=None) -> list:
    """
    utility function to clean column names


    :param ls_col_names: list of strings to modify
    :param chars_to_remove: default: ['\.', ' ', '-'] replaced with _
    :return: list of strings, modified
    """

    if chars_to_remove is None:
        chars_to_remove = ['\.', ' ', '-']

    ls_col_names = ls_col_names.str.lower()
    for c in chars_to_remove:
        ls_col_names = ls_col_names.str.replace(c, '_')

    return ls_col_names
