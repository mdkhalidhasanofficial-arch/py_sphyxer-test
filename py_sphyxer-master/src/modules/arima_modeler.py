
import datetime
import logging
from pathlib import Path

import math
import numpy as np
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import pmdarima as pm
import statsmodels.tools.sm_exceptions as sme
import statsmodels.tsa.arima.model as am

from matplotlib import pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

from cs_utils import log_machine
from ticker.tkr import Tckr


@log_machine
def baseline_arima(df: pd.DataFrame, n_train : int, response: str, elem: str,
                   acf_pacf_plot: bool, autofit: bool, pdq : tuple, n_fcst : int):
    """

    :param df: dataframe that contains, at least, the response variable to be modeled
    :param n_train:
    :param response:
    :param elem:
    :param acf_pacf_plot:
    :return:
    """

    logger = logging.getLogger(__name__)

    # ... set as default array to fit, will be overridden if autofit=True
    ar_time_series_to_fit = np.asarray(df[response].astype('float32'))
    (p, d, q) = pdq
    n_steps = n_fcst

    lx_param = {}
    autofit_model = None
    ls_forecast = [np.nan] * n_steps
    rmse = None
    pct_rmse = None

    if response in df.columns:
        col0 = df.columns.tolist()[0]
        df_plot = df[[col0] + [response]]
    else:
        logger.error("*** requested response variable not in dataframe")
        return

    if acf_pacf_plot:
        # Original Series
        fig, axes = plt.subplots(3, 3, sharex=False)
        axes[0, 0].plot(df_plot[response])
        axes[0, 0].set_title('Original Series')
        plot_acf(df_plot[response], ax=axes[0, 1], lags=10)
        plot_pacf(df_plot[response], ax=axes[0, 2], lags=20)

        # 1st Differencing
        axes[1, 0].plot(df_plot[response].diff())
        axes[1, 0].set_title('1st Order Differencing')
        plot_acf(df_plot[response].diff().dropna(), ax=axes[1, 1], lags=10)
        plot_pacf(df_plot[response].diff().dropna(), ax=axes[1, 2], lags=20)

        # 2nd Differencing
        axes[2, 0].plot(df_plot[response].diff().diff())
        axes[2, 0].set_title('2nd Order Differencing')
        plot_acf(df_plot[response].diff().diff().dropna(), ax=axes[2, 1], lags=10)
        plot_pacf(df_plot[response].diff().diff().dropna(), ax=axes[2, 2], lags=20)

    # ... use autofit for arima parameters on train series

    if autofit:

        # ... split series into train and test sequences
        df_train = df.head(n_train).copy()
        n_test = len(df) - n_train
        df_test = df.tail(n_test)
        df_test = df_test.copy()

        autofit_model = pm.auto_arima(df_train[response], start_p=1, start_q=1,
                              test='adf',  # use adftest to find optimal 'd'
                              max_p=7,
                              max_q=7,  # maximum p and q
                              m=1,  # frequency of series
                              d=None,  # let model determine 'd'
                              seasonal=False,  # No Seasonality
                              start_P=0,
                              D=0,
                              trace=False, # print progress during autofit
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)

        logger.info(elem + str(autofit_model.order))

        # ... fit selected model on observed training data

        (p, d, q) = (int(autofit_model.order[0]), int(autofit_model.order[1]), int(autofit_model.order[2]))
        ar_time_series_to_fit = np.asarray(df_train[response].astype('float32'))
        n_steps = n_test

    try:
        the_model = am.ARIMA(ar_time_series_to_fit, order=(p, d, q), trend='t')
        fit_model = the_model.fit()

        # ... store model params in dict
        ls_param_name = fit_model.param_names
        ls_param = fit_model.params
        for k, v in zip(ls_param_name, ls_param):
            lx_param[k] = v

        # ... forecast into test period
        test_forecast = fit_model.get_forecast(steps=n_steps)
        ls_forecast = test_forecast.prediction_results._forecasts.reshape(n_steps,).tolist()

        if autofit:
            # ... some error metrics on test forecasts vs observed in test period time series
            df_test['error'] = [(y - yhat) for (y, yhat) in zip(df_test[response], ls_forecast)]
            df_test['pct_error'] = [(e / y) for (e, y) in zip(df_test['error'], df_test[response])]
            df_test['sq_error'] = [x * x for x in df_test['error']]
            df_test['pct_sq_error'] = [x * x for x in df_test['pct_error']]
            rmse = math.sqrt(df_test['sq_error'].mean())
            pct_rmse = math.sqrt(df_test['pct_sq_error'].mean())

    except np.linalg.LinAlgError as lae:
        logger.error(str(lae))
    except ValueError as ve:
        logger.error(str(ve))
    except UserWarning as uw:
        logger.info(str(uw))
    except sme.ConvergenceWarning as cw:
        logger.info(str(cw))

    return autofit_model, lx_param, ls_forecast, rmse, pct_rmse


def forecast(config:dict, tckr: Tckr) -> pd.DataFrame:
    # ... forecast for future
    # ... choose a model for each region, category based
    # ...   -  preference towards repeated (p,d,q), lower errors, and more recent periods

    logger = logging.getLogger(__name__)
    logger.info('Begin forecast modeling ...')

    # .. read in pdq values from csv file
    # ... assumes selected pdq values stored in csv file
    f_pdq = Path(config['input']['model_pdq'])
    df_model_pdq = dr.get_model_pdq(f_pdq)

    p = 1
    d = 1
    q = 1

    element = tckr.symbl

    n_fcst = config['ntest']
    n_train = config['ntrain']

    df_forecast_model = tckr.df_price.tail(n_train).copy()

    # ... add rows for future forecast dates
    start_date = config['end_date'] + datetime.timedelta(days=1)
    end_date = config['end_date'] + datetime.timedelta(days=n_fcst)
    forecast_dates = pd.date_range(start_date, end_date, freq='d')

    df_fcst_dates = pd.DataFrame(forecast_dates.to_series(),
                                 columns=['date'])

    df_fcst = pd.concat([df_forecast_model, df_fcst_dates])

    # ... list for the sequence of test period forecasts
    ls_preds = [np.nan] * n_train

    the_model, lx_param, ls_prediction, rmse, pct_rmse = baseline_arima(df_forecast_model,
                                                                        n_train=n_train,
                                                                        response='departure_count',
                                                                        elem=element,
                                                                        acf_pacf_plot=False,
                                                                        autofit=False,
                                                                        pdq=(p, d, q),
                                                                        n_fcst=n_fcst)
    ls_preds += ls_prediction

    df_fcst['predicted'] = ls_preds

    # ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # ... record the forecasts to csv file
    # ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    fpath = config['output']['report_path']
    df_fcst.to_csv(fpath, index=False)

    return df_fcst


