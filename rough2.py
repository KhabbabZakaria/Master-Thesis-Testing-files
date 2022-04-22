import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import pandas as pd
import matplotlib.dates as mpl_dates
from cProfile import label
from turtle import color
from data import *
from actorcritic import *
from parameters import *
import pandas as pd
import csv
import requests
import time
import os
import math  
import itertools
import numpy as np
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))


current_dir = os.getcwd()
ref_folder = os.path.join(current_dir, 'preprocessed_dir')
files = [name for name in os.listdir(ref_folder) if name!= 'vwap calculation.py']


file = files[ref_stock]


plt.style.use('ggplot')

# Extracting Data for plotting
df = pd.read_csv(os.path.join(ref_folder, file))
df = df.iloc[-T*100:]
df = df.iloc[:T]


def candlestick(t, o, h, l, c):
    #plt.figure(figsize=(12,4))
    color = ["green" if close_price < open_price else "red" for close_price, open_price in zip(c, o)]
    plt.plot(x=t, height=np.abs(o-c), bottom=np.min((o,c), axis=0), width=0.6, color=color)
    plt.plot(x=t, height=h-l, bottom=l, width=1, color=color)

df = df.iloc[::-1]

df.index = pd.DatetimeIndex(df['time'])
import mplfinance as mpf
mpf.plot(df,type='candle',mav=(3,6,9),volume=True)