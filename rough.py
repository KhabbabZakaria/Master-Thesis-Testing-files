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


os.chdir(os.path.dirname(os.path.abspath(__file__)))


current_dir = os.getcwd()
ref_folder = os.path.join(current_dir, 'preprocessed_dir')
files = [name for name in os.listdir(ref_folder) if name!= 'vwap calculation.py']

check_max = True
check_max_minus_corresponding = False
#check_avgbest_minus_corresponding = False

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
        list1.remove(max1)
        final_list.append(max1)
          
    return final_list


vwapfinal = []
modelfinal = []
for a in range(20):
    #a = 1
    df = pd.read_csv(os.path.join(ref_folder, file))
    df = df.iloc[:T*50]

    volume = df['volume'].iloc[T*a:T*(a+1)].tolist()
    Q = sum(volume)*0.01



    price = df['vwap'].iloc[T*a:T*(a+1)].tolist()


    volumes = df['volume'].iloc[T*a:T*(a+1)]
    volume_list = volumes.values.tolist()


    weight_vwap_list = []
    for i in range(len(volume_list)):
        weight_vwap_list.append(volume_list[i]/sum(volume_list))


    vwaplist = []
    for i in range(len(df.iloc[T*a:T*(a+1)])):
        q = Q*weight_vwap_list[i]*price[i]
        vwaplist.append(q)

    index = vwaplist.index(max(vwaplist))

    vwaplist2 = []
    for n in range(len(vwaplist)):
        vwaplist2.append(volume_list[n] - vwaplist[n])
        


    if check_max == True:
        vwapfinal.append(max(vwaplist))
        y_label = 'Maximum Volume Sold of all the Minutes on a Day'
    elif check_max_minus_corresponding == True:
        vwapfinal.append(volume_list[index] - vwaplist[index])
        y_label = 'Corresponding Market Sale - Sale by Strategy on Most Extreme Case'
    else:
        vwapfinal.append(sum(Nmaxelements(vwaplist2, 39))/39)
        y_label = 'Average(Market Sale -  Sale by Strategy) on 10% of Extreme Cases'
    #vwapfinal.append(volume_list[index] - vwaplist[index])

    #vwapfinal.append(sum(Nmaxelements(vwaplist, 10))/10)

    modellist = []
    for k in range(1,2):
        PATH = 'seed_' + str(k) + '_model.pt'
        #print(PATH)

        agent = Agent(model, is_eval=True, model_name=PATH)
        env.reset()
        state, _ = getState(all_stressed_test_list[-(a+1)], all_stressed_test_vwap_list[-(a+1)], env.time, state_size)
        for j in range(state_size, len((all_stressed_test_list[15]))):
            already_bought_tensor = torch.Tensor(env.already_bought_list)
            left_time = T - env.time
            left_time_tensor = torch.Tensor(list(range(left_time+1,left_time+1+state_size)))
            private_input = torch.stack((already_bought_tensor, left_time_tensor)).reshape(state_size, 2)
            private_input = private_input.to(device)
            state = state.to(device)
            action, _ = agent.act(private_input, state)
            modellist.append((action*Q*price[j]).item())

    index = modellist.index(max(modellist))

    modellist2 = []
    for n in range(len(modellist)):
        modellist2.append(volume_list[n] - modellist[n])
    modelfinal.append(sum(Nmaxelements(modellist2, 39))/39)

    #modelfinal.append(volume_list[index] - modellist[index])
    #modelfinal.append(sum(Nmaxelements(modellist, 10))/10)
print('.......................................................................................')

'''
vwapmax = max(vwapfinal)
modelmax = max(modelfinal)

print(vwapmax, modelmax)'''

import matplotlib.pyplot as plt
plt.plot(vwapfinal, label = 'vwap', color='blue', linestyle='dashed', marker='o')
plt.plot(modelfinal, label = 'DRL', color='red', linestyle='dashed', marker='o')
plt.legend()
plt.ylabel(y_label)
plt.xlabel('20 Trading Days')
plt.xticks([], minor=True)
plt.title('AAPL during Inflation+War')
plt.show()