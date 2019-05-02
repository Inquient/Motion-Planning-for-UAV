#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from math import sqrt, cos, sin, atan2
import time
import pandas as pd
#import matplotlib.pyplot as plt 
#from mpl_toolkits.mplot3d import Axes3D

def dist(p1,p2):
    return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])+(p1[2]-p2[2])*(p1[2]-p2[2]))

def step_from_to(p1,p2):
    if dist(p1,p2) < EPSILON:
        return p2
    else:
        theta1 = atan2(p2[1]-p1[1],p2[0]-p1[0])
        theta2 = atan2(p2[2]-p1[2],p2[0]-p1[0])
        p1 = p1[0] + EPSILON*cos(theta1), p1[1] + EPSILON*sin(theta1), p1[2] + EPSILON*sin(theta2)
        return p1

EPSILON = 7.0
NUMNODES = 10000

start = (0,0,0)
goal = (90, 90, 90)

stat = []
times = []
lengths = []
experiments = 100

for x in range(0, experiments):
    t = time.time()
    
    random_tree = {}
    current = 0

    nodes = []
    nodes.append(start)
    for i in range(NUMNODES):
        rand = random.random()*100, random.random()*100, random.random()*100
            
        cur_node = nodes[0]
        for p in nodes:
            if dist(p,rand) < dist(cur_node,rand):
                cur_node = p
        if dist(cur_node, goal) < EPSILON:
            random_tree[goal] = cur_node
            current = cur_node
            stat.append(i)
            print('Experiment ', x, ' finished. Iterations= ', i)
            times.append(time.time()-t)
            break
        newnode = step_from_to(cur_node,rand)
        nodes.append(newnode)
        random_tree[newnode] = cur_node
        
    path = [current]
    while current != start:
            current = random_tree[current]
            path.append(current)
     
    path_length = 0
    for i in range(0, len(path)-1):
        path_length += dist(path[i], path[i+1])
    lengths.append(path_length)

print("Количество узлов, для нахождения пути")
print("Среднее = ", np.average(stat))
print("Min = ", np.min(stat))
print("Max = ", np.max(stat))
print("SD = ", np.std(stat))

print("Время, для нахождения пути")
print("Среднее = ", np.average(times))
print("Min = ", np.min(times))
print("Max = ", np.max(times))
print("SD = ", np.std(times))

print("Длинна итогового пути")
print("Среднее = ", np.average(lengths))
print("Min = ", np.min(lengths))
print("Max = ", np.max(lengths))
print("SD = ", np.std(lengths))

df = pd.DataFrame(
        {'stat' : stat,
         'times' : times,
         'lengths' : lengths,  })
    
df.to_csv('3D-RRT stats.csv')
    


