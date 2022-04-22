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

df = pd.read_csv(os.path.join(ref_folder, file))
df = df.iloc[-T*100:]
df = df.iloc[:T]
df = df.iloc[::-1]
print(len(df))
Q = sum(df['volume'].tolist())*0.01
print(Q)
volume_list = df['volume'].tolist()

price = df['vwap'].tolist()

volume_list2 = []
for i in range(len(volume_list)):
    volume_list2.append(volume_list[i]*price[i])

twap_list  = []
for i in range(len(df)):
    twap_list.append(Q/T*price[i])

weight_vwap_list = []
for i in range(len(volume_list)):
    weight_vwap_list.append(volume_list[i]/sum(volume_list))

vwap_list = []
for i in range(len(df)):
    q = Q*weight_vwap_list[i]
    vwap_list.append(q*price[i])


model_list = []
for k in range(1,2):
    PATH = 'seed_' + str(k) + '_model.pt'
    #print(PATH)
    agent = Agent(model, is_eval=True, model_name=PATH)
    env.reset()
    state, _ = getState(all_stressed_test_list[-1], all_test_vwap_list[-1], env.time, state_size)
    for j in range(state_size, len((all_stressed_test_list[15]))):
        if j%3==0:
            already_bought_tensor = torch.Tensor(env.already_bought_list)
            left_time = T - env.time
            left_time_tensor = torch.Tensor(list(range(left_time+1,left_time+1+state_size)))
            private_input = torch.stack((already_bought_tensor, left_time_tensor)).reshape(state_size, 2)
            private_input = private_input.to(device)
            state = state.to(device)
            action, _ = agent.act(private_input, state)
            if action*Q*price[j] > 1e6:
                model_list.append((action*Q*price[j]).item())
                model_list.append((action*Q*price[j]).item())
                model_list.append((action*Q*price[j]).item())

            else:
                model_list.append((action*Q*price[j]).item())
                model_list.append((action*Q*price[j]).item())
                model_list.append((action*Q*price[j]).item())


print(df)

#model_list = [x - 500000 for x in model_list]
model_list2 = modellist




x = list(range(len(volume_list)))


plt.plot(twap_list, label = 'Trade by TWAP', color = 'b')
plt.fill_between(
    x, volume_list2, alpha=0.25, 
    label="Total trade in the market"
)
plt.grid(alpha=0.2)

plt.ylim(0,3000000)
plt.xlabel('Minute of the Day')
plt.ylabel('Volume to Sell in USD')
plt.legend()
plt.title('Division of the Order in TWAP Strategy')
plt.show()



x1 = list(range(len(vwap_list)))
coef_vwap = np.polyfit(x1,vwap_list,5)
poly1d_fn_vwap = np.poly1d(coef_vwap) 

plt.plot(x1, poly1d_fn_vwap(x1), 'b-', label = 'Regression Line')
plt.plot(vwap_list, label = 'Trade by VWAP', color = '0.8')
plt.fill_between(
    x, volume_list2, alpha=0.25, 
    label="Total trade in the market"
)
plt.grid(alpha=0.2)

plt.ylim(0,3000000)
plt.xlabel('Minute of the Day')
plt.ylabel('Volume to Sell in USD')
plt.legend()
plt.title('Division of the Order in VWAP Strategy')
plt.show()



x1 = list(range(len(model_list2)))
coef_vwap = np.polyfit(x1,model_list2,5)
poly1d_fn_vwap = np.poly1d(coef_vwap) 

plt.plot(x1, poly1d_fn_vwap(x1), 'b-', label = 'Regression Line')
plt.plot(model_list2, label = 'Trade by DRL', color = '0.8')
plt.fill_between(
    x, volume_list2, alpha=0.25, 
    label="Total trade in the market"
)
plt.grid(alpha=0.2)

plt.ylim(0,3000000)
plt.xlabel('Minute of the Day')
plt.ylabel('Volume to Sell in USD')
plt.legend()
plt.title('Division of the Order in DRL Strategy')
plt.show()
