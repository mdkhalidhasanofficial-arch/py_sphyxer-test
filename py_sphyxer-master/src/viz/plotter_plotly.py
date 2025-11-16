# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...   utility methods to act on post processing files
# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...   imports
# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


import sys
from os.path import dirname, join, abspath
import pandas as pd
from typing import Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import locale

from cs_utils import log_machine

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
locale.setlocale(locale.LC_ALL, '')
sys.path.insert(0, '../src/')

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... plot format options
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

colors = {
    'big_background': '#1E2B3C',
    'paper_background': '#2f435e',
    'paper_background20': 'rgba(47,43,94,5)',
    'plot_background': '#909fb2',
    'plot_grid': '#c1c9d3',
    #    'plot_grid' : '#657891',
    'text': '#ffe45b',
    'text20': 'rgba(255,228,91,20)',
    'text_tbl': '#445161',
    'line_color': '#b467af'
}
line_width = [1, 2, 3, 4]

text_style = dict(
    color='#ffe45b',
    fontFamily='tisa',
    fontWeight=300,
    textAlign='center')


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... time series observed and predict values plots - single run id
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@log_machine
def make_time_series_plot(df_plot: pd.DataFrame, symbl: str, ls_cols: list) -> Tuple[go.Figure, dict]:
    """
    produce time series plot, displaying observed and multiple model forecasts, along with the
    explanatory variables. plot object (dash) can be displayed in html as dynamically manipulable
    time series  plot

    :param df_plot: data frame containing merged processed input and predictions, single model part id
    :param ptype: string, default = 'fcst' - used for plot filename
    :return: fig_ts_grid, cnfg_ts_grid - plot and config objects, (plotly objects)
    """

    # ... some plot element characteristics

    marker_props = {'size': 4,
                    'opacity': 0.6,
                    'line': {'color': colors['paper_background20'], 'width': 1}
                    }

    # ... set up plot rows and columns

    plot_height = 600
    plot_width = 900

    main_title = symbl + ' - time history'

    fig_ts_grid = make_subplots(rows=1,
                                cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.03)

    ls_df_cols = df_plot.columns.tolist()
    for this_y in ls_cols:
        if this_y in ls_df_cols:
            fig_ts_grid.add_trace(go.Scatter(x=df_plot['date'],
                                             y=df_plot[this_y],
                                             name=this_y,
                                             mode='lines+markers',
                                             marker=marker_props))

    # ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # ... finish formats and title updates
    # ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    fig_ts_grid.update_layout(
        height=plot_height,
        title_text=main_title,
        font_color=colors['text'],
        plot_bgcolor=colors['plot_background'],
        paper_bgcolor=colors['paper_background'],
        yaxis_tickformat='.1f',
        hovermode="x unified")

    fig_ts_grid.update_xaxes(showline=True, linewidth=1, linecolor=colors['plot_grid'])
    fig_ts_grid.update_yaxes(showline=True, linewidth=1, linecolor=colors['plot_grid'])
    fig_ts_grid.update_xaxes(showgrid=True, gridwidth=1, gridcolor=colors['plot_grid'])
    fig_ts_grid.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['plot_grid'])

    png_name = symbl

    cnfg_ts_grid = {
        'toImageButtonOptions': {
            'format': 'png',  # one of png, svg, jpeg, webp
            #            'filename': png_name,
            'height': plot_height,
            'width': plot_width,
            'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
        }
    }

    return fig_ts_grid, cnfg_ts_grid
