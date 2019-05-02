#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from math import sqrt, cos, sin, atan2
import time
import pandas as pd

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
NUMNODES = 1000

start = (0,0,0)
goal = (90, 90, 90)

progress = np.arange(0.0 ,1.0 , 0.1)

stat = []
times = []
lengths = []
experiments = 100

for x in range(0, experiments):
    t = time.time()
    
    start_nodes = []
    goal_nodes = []
    
    start_tree = {}
    goal_tree = {}
    current1 = 0
    current2 = 0

    start_nodes.append(start)
    goal_nodes.append(goal)
        
    for i in range(NUMNODES):
        flag = True
        rand1 = random.random()*100, random.random()*100, random.random()*100
        cur_node1 = start_nodes[0]
        for p in start_nodes:
            if dist(p,rand1) < dist(cur_node1,rand1):
                cur_node1 = p
        newnode = step_from_to(cur_node1,rand1)
        start_nodes.append(newnode)
        start_tree[newnode] = cur_node1
        
        rand2 = random.random()*100, random.random()*100, random.random()*100
        cur_node2 = goal_nodes[0]
        for p in goal_nodes:
            if dist(p,rand2) < dist(cur_node2,rand2):
                cur_node2 = p
        newnode = step_from_to(cur_node2,rand2)
        goal_nodes.append(newnode)
        goal_tree[newnode] = cur_node2        
        
        for a in start_nodes:
            for b in goal_nodes:
                if dist(a, b) < EPSILON:
                    current1 = a
                    current2 = b
                    stat.append(i*2)
                    #print('Experiment ', x, ' finished. Iterations= ', i*2)
                    flag = False
                    times.append(time.time()-t)
                    break
            if not flag:
                break
        if not flag:
            break
        
    start_path = [current1]
    while current1 != start:
            current1 = start_tree[current1]
            start_path.append(current1)
         
    goal_path = [current2]
    while current2 != goal:
            current2 = goal_tree[current2]
            goal_path.append(current2)
    
    start_path.reverse()
    full_path = start_path + goal_path
    
    path_length = 0
    for i in range(0, len(full_path)-1):
        path_length += dist(full_path[i], full_path[i+1])
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
    
df.to_csv('3D-RRT-connect stats.csv')
    
    