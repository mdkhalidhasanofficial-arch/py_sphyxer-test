
"""

"""
import os
import sys
import platform
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import constants as c
import cs_utils as cu
import modeler as mo
from cs_utils import Timer, replace_text_line
from viz.plotter_plotly import make_time_series_plot
from ticker.tkr import Tckr
from portfolio.portfolio import Portfolio
from trade.trader import Trader

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# ... share some system and package info
print(platform.platform())
print("python", sys.version)
print("numpy", np.__version__)
print("pandas", pd.__version__)

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... main() routine
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


if __name__ == "__main__":

    config = cu.setup_this_run(sys.argv)
    logger = cu.setup_logger(sys.argv, config)

    fpath = Path(config['output']['plot_dir'])
    Path(fpath).mkdir(parents=True, exist_ok=True)

    # git_repo = u.get_git_repo_info()

    # ... report version executing
    logger.critical("Executing : %s", os.path.split(sys.argv[0])[1])
    logger.critical("Platform : %s", platform.platform())
    logger.critical("python : %s", sys.version.replace('\n', ''))
    logger.critical("Executing from : %s", os.getcwd())
    logger.critical("pandas : %s", pd.__version__)
    logger.critical("numpy  : %s", np.__version__)
    # logger.critical("git branch : " + git_repo.get('branch'))
    # logger.critical("git commit : " + git_repo.get('sha'))
    # logger.critical("git commit date : " + git_repo.get('commit_date'))
    logger.critical("Logging level : %s", config['logging_level'])
    logger.critical(" ")

    # u.echo_python_packages(config)

    logger.critical('start')

    # set timer for overall process timing
    tic = Timer()
    tic.start()

    # ... initialize starting portfolio
    df_port = pd.DataFrame({'date': [datetime.datetime(2020, 1, 1), datetime.datetime(2022, 1, 1)],
                            'tckr_symbol': ['GE', 'MSFT'],
                            'qty': [100, 200],
                            'fee': [1, 2]
                            }, index=[0, 1])
    portfolio = Portfolio(config, df_port)
    logger.info(f'Initial portfolio value: {portfolio.value}')

    # ... initiate Trader
    socrates = Trader('socrates', config, portfolio, None)

    # ... run model and predictions for sequential time periods
    for end_month in range(6, 13):

        end_date = '2024-' + f'{end_month:02d}' + '-01'
        config['end_date'] = end_date

        # ... do the modeling
        status = mo.orchestrate_modeler(config)

        # ... evaluate forecasts by trader for buy/sell decisions

        # ... transact per trader decisions

        # ... update / trend portfolio value

        logger.info('')
        logger.info(f'*** {end_date} completed; status = {status} ***')
        logger.info('')

    # ... porfolio values
    portfolio.get_current_holding(datetime.datetime.now())
    logger.info(f'Current portfolio value: {portfolio.value}')

    # log final message
    logger_msg = 'overall execution time = %.2f' % tic.stop(False)
    logger.critical('complete - %s', logger_msg)

    print('execution complete - %.2f seconds' % tic.stop())

# ... end of script
