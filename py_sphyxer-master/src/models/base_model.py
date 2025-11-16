
import pandas as pd
import numpy as np


class BaseModel:
    """
    Base model class
    """

    def __init__(self):
        """
        required parameters to instantiate base model:

        lx_param: (dict) record model parameters
        the_model: (model_object)
        fit_model: the model fit on current data set
        df_model_data: (pd.DataFrame) - standard dataframe of raw data
        dq, fq (dict) : record status of data quality, model fit quality evaluations

        """

        self.lx_param = {}

        self.the_model = None
        self.fit_model = None
        self.df_model_data = pd.DataFrame()

        self.dq = {'dq': np.NaN}
        self.fq = {'fq': np.NaN}


class BaseModelError(Exception):
    """
    BaseModel custom error exception class
    """

    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload

    def __str__(self):
        return str(self.message)
