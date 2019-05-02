#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from math import sqrt, cos, sin, atan2
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

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
NUMNODES = 100

start = (0,0,0)
goal = (90, 90, 90)

progress = np.arange(0.0 ,1.0 , 0.1)

fig= plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(start[0], start[1], start[2], color = 'k')
ax.scatter(goal[0], goal[1], goal[2], color = 'r')

start_nodes = []
goal_nodes = []

start_tree = {}
goal_tree = {}
current1 = 0
current2 = 0

start_nodes.append(start)
goal_nodes.append(goal)

flag = True

for i in range(NUMNODES):
    rand1 = random.random()*100, random.random()*100, random.random()*100
    cur_node1 = start_nodes[0]
    for p in start_nodes:
        if dist(p,rand1) < dist(cur_node1,rand1):
            cur_node1 = p
    newnode = step_from_to(cur_node1,rand1)
    start_nodes.append(newnode)
    start_tree[newnode] = cur_node1
    plt.plot([cur_node1[0], newnode[0]], [cur_node1[1], newnode[1]], [cur_node1[2], newnode[2]], color = 'b')
    
    rand2 = random.random()*100, random.random()*100, random.random()*100
    cur_node2 = goal_nodes[0]
    for p in goal_nodes:
        if dist(p,rand2) < dist(cur_node2,rand2):
            cur_node2 = p
    newnode = step_from_to(cur_node2,rand2)
    goal_nodes.append(newnode)
    goal_tree[newnode] = cur_node2
    plt.plot([cur_node2[0], newnode[0]], [cur_node2[1], newnode[1]], [cur_node2[2], newnode[2]], color = 'r')
    
    
    for a in start_nodes:
        for b in goal_nodes:
            if dist(a, b) < EPSILON:
                current1 = a
                current2 = b
                plt.plot([a[0], b[0]], [a[1], b[1]],  [a[2], b[2]], color = 'k', linewidth=7)
                print(i)
                flag = False
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
    plt.plot([full_path[i][0], full_path[i+1][0]], [full_path[i][1], full_path[i+1][1]], [full_path[i][2], full_path[i+1][2]], color = 'g')

print(path_length)
plt.show()