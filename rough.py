from cProfile import label
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
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import yfinance as yf


os.chdir(os.path.dirname(os.path.abspath(__file__)))


current_dir = os.getcwd()
ref_folder = os.path.join(current_dir, 'preprocessed_dir')
files = [name for name in os.listdir(ref_folder) if name!= 'vwap calculation.py']




def calc_twap(df_csv):
    n = len(df_csv) - 1
    price_sum = 0.0
    for i in range(1, n + 1):
        high_price = df_csv['high'].iloc[i]
        low_price = df_csv['low'].iloc[i]
        close = df_csv['close'].iloc[i]
        price = (high_price + low_price + close) / 3
        price_sum += price

    return price_sum / n


def calc_vwap(df_csv):
    n = len(df_csv) - 1
    total_sum = 0.0
    volume_sum = 0
    for i in range(1, n + 1):
        high_price = df_csv['high'].iloc[i]
        low_price = df_csv['low'].iloc[i]
        close = df_csv['close'].iloc[i]
        price = (high_price + low_price + close) / 3
        volume = df_csv['volume'].iloc[i]
        total_sum += price * volume
        volume_sum += volume

    return total_sum / volume_sum



file = files[ref_stock]
#print('........................................................')
#print(file)
def Nmaxelements(list1, N):
    final_list = []
  
    for i in range(0, N): 
        max1 = -10000000
          
        for j in range(len(list1)):     
            if list1[j] > max1:
                max1 = list1[j]
        print(max1)     
        list1.remove(max1)
        final_list.append(max1)
          
    return final_list



a = 0
df = pd.read_csv(os.path.join(ref_folder, file))
df = df.iloc[-T*100:]

volume = df['volume'].iloc[T*a:T*(a+1)].tolist()
print(volume)
Q = sum(volume)*0.01
print(Q)
price = df['vwap'].iloc[T*a:T*(a+1)].tolist()
twap = [Q/T]*T

print(df)







'''fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(price, 'g--', label = 'Price')
ax2.plot(twap, 'b-')

ax1.set_xlabel('Minute of the Day')
ax1.set_ylabel('Price of the Stock in USD', color='g')
ax2.set_ylabel('How much stocks we need to sell', color='b')
plt.legend()
plt.title('AMZN on Date: 2020-08-12 using TWAP')


plt.show()'''

import plotly.graph_objects as go

import pandas as pd
from datetime import datetime

df = df[:T]

fig = go.Figure(data=[go.Candlestick(x=df['time'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'])])

fig.show()