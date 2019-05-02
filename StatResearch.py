#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns

rrt_data = pd.read_csv('3D-RRT stats.csv')
connect_data = pd.read_csv('3D-RRT-connect stats.csv')


data = pd.concat([rrt_data, connect_data], axis=1)
data.columns = ['num','rrt_stat','rrt_times','rrt_lengths','num','connect_stat','connect_times','connect_lengths']
data.drop(['num','num'],inplace=True, axis=1)
print(data.head())

sns.set(style="whitegrid")
ax = sns.boxplot(x=data['rrt_stat'], orient='v')
ax = sns.swarmplot(x=data['rrt_stat'], orient='v')

