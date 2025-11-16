

import logging

import numpy as np

import pmdarima as pm
import statsmodels.tools.sm_exceptions as sme
import statsmodels.tsa.arima.model as am

import data_readers
from models.base_model import BaseModel
from cs_utils import log_machine
import data_readers as dr

logger = logging.getLogger(__name__)


class ARIMA(BaseModel):
    """
    ARIMA model, inherits from BaseModel
    """

    def __init__(self):

        super().__init__()

        self.autofit = True
        self.pdq = (0, 0, 0)

    def prepare_data(self, tkr: str):
        """
        checks dates for consistency with start and end requested dates
        trims dataframe to only contain records in model period
        updates self.df_model_data accordingly

        :param tkr: str
        :return: None
        """

        if tkr.df_price['date'].min() > tkr.first_date:
            err_msg = 'Tkr price history does not precede required model start date'
            logger.warning(err_msg)
            tkr.df_price = dr.get_ticker_data(tkr.symbl, tkr.first_date, tkr.last_date)

        if tkr.df_price['date'].max() < tkr.last_date:
            err_msg = f"Tkr history does not extend to required end date {tkr.df_price['date'].max()} | {tkr.last_date}"
            logger.warning(err_msg)
            tkr.df_price = dr.get_ticker_data(tkr.symbl, tkr.first_date, tkr.last_date)

        df_model_data = tkr.df_price[tkr.df_price['date'] >= tkr.first_date]
        df_model_data = df_model_data[df_model_data['date'] <= tkr.last_date]
        df_model_data.reset_index(inplace=True, drop=True)

        self.df_model_data = df_model_data

    @log_machine
    def tune(self, symbl: str, response: str, n_train: int, n_test: int) -> None:
        """
        autofit tuning for ARIMA paramaters, updates self.pdq of mdoel object

        :param symbl: (str) label - used for logging purposes only
        :param df: (pd.DataFrame) - time-series to be modeled, outcome of model.prepare_data
        :param response: (str) - column name in df to identify time-series
        :param n_train: (int) - n-train rows in data column for tuning
        :param n_test: (int) - n_test rows in data column for out-of-sample
        :return: None, updates self.pdq with (p,d,q) tuple
        """

        logger = logging.getLogger(__name__)
        logger.info('start')

        df = self.df_model_data

        if not response in df.columns.tolist():
            logger.warning(f'response column {response} not found for symbol: {symbl}')
            raise KeyError

        if self.autofit:
            # ... split series into train and test sequences

            df_model = df.tail(n_train + n_test)

            try:
                autofit_model = pm.auto_arima(df_model[response], start_p=1, start_q=1,
                                              test='adf',  # use adftest to find optimal 'd'
                                              max_p=7,
                                              max_q=7,  # maximum p and q
                                              m=1,  # frequency of series
                                              d=None,  # let model determine 'd'
                                              seasonal=False,  # No Seasonality
                                              start_P=0,
                                              D=0,
                                              trace=False,  # print progress during autofit
                                              error_action='ignore',
                                              suppress_warnings=True,
                                              stepwise=True,
                                              out_of_sample_size=n_test,
                                              scoring='mse')

                logger.info(symbl + str(autofit_model.order))
                self.pdq = autofit_model.order

            except ValueError as ve:
                logger.warning('Likley stationarity issue with model data: %s - %s', (symbl, str(ve)))

            # ... fit selected model on observed training data

        else:
            logger.error('autofit is the only currently enabled option for tune()')
            raise NotImplementedError

        return None

    @log_machine
    def fit(self, response: str) -> None:
        """
        fits arima model using (p, d, q) from self.pdq

        :param df:
        :param response:
        :return:
        """
        logger = logging.getLogger(__name__)
        logger.info('start')

        df = self.df_model_data

        if response not in df.columns.tolist():
            logger.error('Response column: %s not found in dataframe' % response)
            raise KeyError

        ARIMAFitErrors = (np.linalg.LinAlgError, ValueError)
        ARIMAFitWarnings = (UserWarning, sme.ConvergenceWarning)

        (p, d, q) = self.pdq
        ar_time_series_to_fit = np.asarray(df[response].astype('float32'))

        try:
            self.the_model = am.ARIMA(ar_time_series_to_fit, order=(p, d, q), trend='t')
            self.fit_model = self.the_model.fit()

            # ... store model params in dict
            ls_param_name = self.fit_model.param_names
            ls_param = self.fit_model.params
            for k, v in zip(ls_param_name, ls_param):
                self.lx_param[k] = v

        except ARIMAFitErrors as afe:
            logger.error(str(afe))
        except ARIMAFitWarnings as afw:
            logger.info(str(afw))

    @log_machine
    def predict(self, n_steps: int):
        """

        :param n_steps: int
        :return:
        """

        logger = logging.getLogger(__name__)
        logger.info('ARIMA predict')

        # ... forecast into test period
        test_forecast = self.fit_model.get_forecast(steps=n_steps)
        ls_forecast = test_forecast.prediction_results.results.forecasts.reshape(n_steps, ).tolist()

        # if autofit:
        #     # ... some error metrics on test forecasts vs observed in test period time series
        #     df_test['error'] = [(y - yhat) for (y, yhat) in zip(df_test[response], ls_forecast)]
        #     df_test['pct_error'] = [(e / y) for (e, y) in zip(df_test['error'], df_test[response])]
        #     df_test['sq_error'] = [x * x for x in df_test['error']]
        #     df_test['pct_sq_error'] = [x * x for x in df_test['pct_error']]
        #     rmse = math.sqrt(df_test['sq_error'].mean())
        #     pct_rmse = math.sqrt(df_test['pct_sq_error'].mean())

        return ls_forecast

    def summary(self):
        pass
