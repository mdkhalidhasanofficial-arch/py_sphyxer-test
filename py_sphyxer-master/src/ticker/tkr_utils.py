
import os
import sys
import datetime
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

import src.constants as c


def get_sp500_symbols():
    df_wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df_sp500 = df_wiki_table[0]

    ls_sp500 = df_sp500['Symbol'].tolist()

    for b in c.BAD_TICKER:
        ls_sp500.remove(b)

    return ls_sp500


def plot_tckr(tckr, df_tckr, f_plot, plot_save=True, n_pts=None):
    fig = plt.figure(figsize=(12, 8))

    x = [datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in df_tckr['date']]
    y = df_tckr['adjclose']

    if n_pts is not None:
        x = x[-n_pts:]
        y = y[-n_pts:]

    ax = plt.gca()
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.YearLocator()
    ax.xaxis.set_major_locator(locator)

    plt.scatter(x, y,
                label='adjclose',
                marker='o',
                color='cornflowerblue',
                s=5)

    plt.title(tckr + ' - historical adjclose price')
    plt.xlabel('date')
    plt.ylabel('Price')
    plt.legend()

    # Text in the x axis will be displayed in 'YYYY-mm' format.
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    # Rotates and right-aligns the x labels so they don't crowd each other.
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    plt.grid(True)

    if plot_save:
        plt.savefig(f_plot)
    plt.close()


def plot_line(df, x, y, title, f_plot, plot_save=False):
    fig = plt.figure(figsize=(12, 8))

    plt.plot(df[x], df[y],
             label=y,
             marker='o',
             color='cornflowerblue')

    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()

    plt.grid(True)

    if plot_save:
        plt.savefig(f_plot)
    plt.close()
