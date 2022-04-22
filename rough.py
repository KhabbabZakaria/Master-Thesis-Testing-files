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


a = 1
df = pd.read_csv(os.path.join(ref_folder, file))
df = df.iloc[:T*50]
print(df.iloc[0])
volume1 = df['volume'].iloc[T*a:T*(a+1)].tolist()
price1 = df['vwap'].iloc[T*a:T*(a+1)].tolist()

products = []
for num1, num2 in zip(volume1, price1):
	products.append(num1 * num2)

volume1 = products
volume1 = volume1[::-1]

a = 2
df = pd.read_csv(os.path.join(ref_folder, file))
df = df.iloc[:T*50]
volume2 = df['volume'].iloc[T*a:T*(a+1)].tolist()
price2 = df['vwap'].iloc[T*a:T*(a+1)].tolist()

products = []
for num1, num2 in zip(volume2, price2):
	products.append(num1 * num2)

volume2 = products

volume2 = volume2[::-1]


x1 = list(range(len(volume1)))

coef = np.polyfit(x1,volume1,6)
poly1d_fn1 = np.poly1d(coef) 

x2 = list(range(len(volume2)))

coef = np.polyfit(x2,volume2,6)
poly1d_fn2 = np.poly1d(coef) 

plt.grid()
plt.plot(x1, poly1d_fn1(x1), color = 'b', label = 'Date: 2022-03-10')
plt.plot(x2, poly1d_fn2(x2), color = '0.8', label = 'Date: 2022-03-09')
plt.legend()
plt.xlabel('Minute of the Day')
plt.ylabel('Volume Trade at the Minute')
plt.title('Volumes of FB that is traded on 2 consecutive days during Inflation+War')
plt.show()
