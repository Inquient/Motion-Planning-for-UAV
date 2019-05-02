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

EPSILON = 12.0
NUMNODES = 10000

start = (0,0,0)
goal = (90, 90, 90)

random_tree = {}
current = 0

progress = np.arange(0.0 ,1.0 , 0.1)

fig= plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(start[0], start[1], start[2], color = 'k')
ax.scatter(goal[0], goal[1], goal[2], color = 'r')

nodes = []

nodes.append(start)
for i in range(NUMNODES):
    rand = random.random()*100, random.random()*100, random.random()*100
        
    cur_node = nodes[0]
    for p in nodes:
        if dist(p,rand) < dist(cur_node,rand):
            cur_node = p
    if dist(cur_node, goal) < EPSILON:
        ax.plot([cur_node[0], goal[0]], [cur_node[1], goal[1]], [cur_node[2], goal[2]], color = 'b')
        random_tree[goal] = cur_node
        current = cur_node
        print(i)
        break
    newnode = step_from_to(cur_node,rand)
    nodes.append(newnode)
    
    random_tree[newnode] = cur_node
    
    ax.plot([cur_node[0], newnode[0]], [cur_node[1], newnode[1]], [cur_node[2], newnode[2]], color = 'b')
    if i/NUMNODES in progress:
        print(i/NUMNODES, 'finished')
    



path = [current]
while current != start:
        current = random_tree[current]
        path.append(current)

path_length = 0
for i in range(0, len(path)-1):
    path_length += dist(path[i], path[i+1])
    plt.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]], [path[i][2], path[i+1][2]], color = 'r')

print(path_length)    
plt.show()



