#Comparing the Price Advantage and probable other results from TWAP, VWAP and DRL models for an entire dataset, for example testset1, testset2, stressed testset, etc

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

#Assumption = we are selling 10**5 units on a day (not most important, can be arbitrary)
Q = 10**5

#twap price calculation
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

#vwap price calculation
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

# ref stock is the index of stock from either Large or Mid or Small Caps or ETFs
file = files[ref_stock]
#print('........................................................')
#print(file)
df = pd.read_csv(os.path.join(ref_folder, file))
df = df.iloc[-T*100:]
#df = df.iloc[-T*150:-T*100]

vwaps = df['vwap']
vwaps_list = vwaps.values.tolist()

volumes = df['volume']
volume_list = volumes.values.tolist()

summed_volume_list = []
for i in range(int(len(volumes)/T)):
    volume = volumes.iloc[i*T:(i+1)*T]
    summed_volume = sum(volume)
    summed_volume_list.append(summed_volume)


summed_volume_list = list(itertools.chain.from_iterable(itertools.repeat(x, T) for x in summed_volume_list))

weight_vwap_list = []
for i in range(len(summed_volume_list)):
    weight_vwap_list.append(volume_list[i]/summed_volume_list[i])


'''summed_vwap_list = []
for i in range(int(len(vwaps)/T)):
    vwap = vwaps.iloc[i*T:(i+1)*T]
    summed_vwap = sum(vwap)
    summed_vwap_list.append(summed_vwap)


summed_vwap_list = list(itertools.chain.from_iterable(itertools.repeat(x, T) for x in summed_vwap_list))

weight_vwap_list = []
for i in range(len(summed_vwap_list)):
    weight_vwap_list.append(vwaps_list[i]/summed_vwap_list[i])'''


############################################
########## calculating price_bar ###########
############################################
#price_bar = sum(vwaps_list)/len(vwaps_list)
#print(price_bar)
price_bar = price_bar_test


twap_price = []
for i in range(int(len(df)/T)):
    price = calc_twap(df.iloc[i*T:(i+1)*T])
    twap_price.append(price)

twap_price = list(itertools.chain.from_iterable(itertools.repeat(x, T) for x in twap_price))
#print('twap_price', twap_price)

vwap_price = []
for i in range(int(len(df)/T)):
    price = calc_vwap(df.iloc[i*T:(i+1)*T])
    vwap_price.append(price)

vwap_price = list(itertools.chain.from_iterable(itertools.repeat(x, T) for x in vwap_price))
#print('vwap_price', vwap_price)
twap_price = np.array(twap_price).reshape(len(all_test_list), T)
vwap_price = np.array(vwap_price).reshape(len(all_test_list), T)
weight_vwap_list = np.array(weight_vwap_list).reshape(len(all_test_list), T)
price_k_bar = twap_price


#######################################################
#########################TWAP##########################
#######################################################
Q = 10
q = Q/T
action = q/Q
total_reward = 0
PA = 0
PA_positve = 0
PA_negitve = 0
for i in range(len(all_test_list)):
    P_bar_strategy = 0 #AEP
    PA_internal = 0
    for j in range(len((all_test_list[i]))):
        price = twap_price[i,j]
        reward = (price/price_bar[i,j] - 1)*action - action**2
        total_reward = total_reward + reward
        P_bar_strategy = P_bar_strategy + (q/Q)*price

    PA = PA + ((P_bar_strategy/price_k_bar[i,0]) - 1)
    PA_internal = PA_internal + ((P_bar_strategy/price_k_bar[i, 0]) - 1)

    if PA_internal <= 1e-10:
        PA_negitve = PA_negitve + 1

    else:
        PA_positve = PA_positve + 1 
    
total_reward = total_reward/(len(all_test_list)*T)
print('TWAP total_reward', total_reward)

PA = 10**2/len(all_test_list) * PA
print('TWAP PA', PA)

try:
    GLR = PA_positve/PA_negitve
    print('TWAP GLR', GLR)
except:
    print('all positive')




#######################################################
#########################VWAP##########################
#######################################################
Q = 10
total_reward = 0
PA = 0
PA_positve = 0
PA_negitve = 0
for i in range(len(all_test_list)):       
    P_bar_strategy = 0 #AEP
    PA_internal = 0
    for j in range(len((all_test_list[i]))):
        q = Q*weight_vwap_list[i,j]
        action = q/Q
        price = vwap_price[i,j]
        reward = (price/price_bar[i,j] - 1)*action - action**2
        total_reward = total_reward + reward
        P_bar_strategy = P_bar_strategy + (q/Q)*price

    PA = PA + ((P_bar_strategy/price_k_bar[i,0]) - 1)
    PA_internal = PA_internal + ((P_bar_strategy/price_k_bar[i,0]) - 1)


    if PA_internal <= 1e-10:
        PA_negitve = PA_negitve + 1

    else:
        PA_positve = PA_positve + 1
    
total_reward = total_reward/(len(all_test_list)*T)
print('VWAP total_reward', total_reward)

PA = 10**2/len(all_test_list) * PA
print('VWAP PA', PA)

try:
    GLR = PA_positve/PA_negitve
    print('VWAP GLR', GLR)
except:
    print('all positive')


########################################################
#########################Model##########################
########################################################

for k in range(1,2):
    PATH = 'seed_' + str(k) + '_model.pt'
    #print(PATH)

    agent = Agent(model, is_eval=True, model_name=PATH)
    Q = 10
    total_reward = 0
    PA = 0
    PA_positve = 0
    PA_negitve = 0
    for i in range(len(all_test_list)):
        P_bar_strategy = 0 #AEP
        PA_internal = 0
        env.reset()
        state, _ = getState(all_test_list[i], all_test_vwap_list[i], env.time, state_size)
        for j in range(state_size, len((all_test_list[i]))):
            price = all_test_vwap_list[i][j]
            
            already_bought_tensor = torch.Tensor(env.already_bought_list)
            left_time = T - env.time
            left_time_tensor = torch.Tensor(list(range(left_time+1,left_time+1+state_size)))
            private_input = torch.stack((already_bought_tensor, left_time_tensor)).reshape(state_size, 2)
            private_input = private_input.to(device)
            state = state.to(device)
            action, _ = agent.act(private_input, state)
            action = action.to(device)
            reward = (price/price_bar[i,j] - 1)*action - action**2
            #print(price, price_bar, reward)
            env.step(action)
            next_state, _ = getState(all_test_list[i], all_test_vwap_list[i], env.time, state_size)
            total_reward = total_reward + reward
            P_bar_strategy = P_bar_strategy + action*price
            state = next_state
            if left_time == 1 and env.already_bought>0:    #no more time left yet target left
                #print('no more time left yet target left')
                total_reward = total_reward - 100
                P_bar_strategy = P_bar_strategy - 100
            if env.done == True:
                break
        PA = PA + ((P_bar_strategy/price_k_bar[i,0]) - 1)
        PA_internal = PA_internal + ((P_bar_strategy/price_k_bar[i,0]) - 1)


        if PA_internal <= 1e-10:
            PA_negitve = PA_negitve + 1

        else:
            PA_positve = PA_positve + 1

    total_reward = total_reward/(len(all_test_list)*T)
    print('Model seed' + str(k) + 'total_reward', total_reward)

    PA = 10**2/len(all_test_list) * PA
    print('Model seed' + str(k) + 'PA', PA)


    try:
        GLR = PA_positve/PA_negitve
        print('Model seed' + str(k) + 'GLR', GLR)
    except:
        print('all are positive')


