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
df = pd.read_csv(os.path.join(ref_folder, file))
df = df.iloc[:T*50]
print(df.iloc[T*15:T*16])

volume = df['volume'].iloc[T*15:T*16].tolist()
Q = sum(volume)*0.01


price = df['vwap'].iloc[T*15:T*16].tolist()


volumes = df['volume'].iloc[T*15:T*16]
volume_list = volumes.values.tolist()


weight_vwap_list = []
for i in range(len(volume_list)):
    weight_vwap_list.append(volume_list[i]/sum(volume_list))


vwaplist = []
for i in range(len(df.iloc[T*15:T*16])):
    q = Q*weight_vwap_list[i]
    vwaplist.append(q)




modellist = []
for k in range(1,2):
    PATH = 'seed_' + str(k) + '_model.pt'
    #print(PATH)

    agent = Agent(model, is_eval=True, model_name=PATH)
    env.reset()
    state, _ = getState(all_stressed_test_list[-16], all_stressed_test_vwap_list[-16], env.time, state_size)

    for j in range(state_size, len((all_stressed_test_list[15]))):
        if j%10==0:
            already_bought_tensor = torch.Tensor(env.already_bought_list)
            left_time = T - env.time
            left_time_tensor = torch.Tensor(list(range(left_time+1,left_time+1+state_size)))
            private_input = torch.stack((already_bought_tensor, left_time_tensor)).reshape(state_size, 2)
            private_input = private_input.to(device)
            state = state.to(device)
            action, _ = agent.act(private_input, state)
            for _ in range(10):
             modellist.append((action*Q).item())

modellist = modellist[::-1]

x = list(range(len(vwaplist)))
coef_vwap = np.polyfit(x,vwaplist,10)
poly1d_fn_vwap = np.poly1d(coef_vwap) 

x = list(range(len(modellist)))
coef_model = np.polyfit(x,modellist,5)
poly1d_fn_model = np.poly1d(coef_model) 



import matplotlib.pyplot as plt

x = list(range(len(volume_list)))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(price, 'g--', label = 'price')
ax2.plot(x, poly1d_fn_vwap(x), 'b-', label = 'vwap')
ax2.plot(x, poly1d_fn_model(x), 'r-', label = 'DRL')
ax2.fill_between(
    x, volume_list, alpha=0.25, 
    label="Total trade in the market"
)

ax1.set_xlabel('Minute of the Day')
ax1.set_ylabel('Price of the Stock in USD', color='g')
ax2.set_ylabel('Stocks to sell at the minute', color='k')
ax2.axis(ymin=0, ymax=10000)
plt.legend()
plt.title('FB on 2022-02-16')
plt.show()
