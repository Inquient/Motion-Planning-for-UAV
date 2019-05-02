#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from math import sqrt, cos, sin, atan2
import matplotlib.pyplot as plt 

def dist(p1,p2):
    return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

def step_from_to(p1,p2):
    if dist(p1,p2) < EPSILON:
        return p2
    else:
        theta = atan2(p2[1]-p1[1],p2[0]-p1[0])
        p1 = p1[0] + EPSILON*cos(theta), p1[1] + EPSILON*sin(theta)
        return p1

EPSILON = 7.0
NUMNODES = 1000

start = (0,0)
goal = (100, 100)


fig, ax = plt.subplots()
plt.scatter(start[0], start[1], color = 'k')
plt.scatter(goal[0], goal[1], color = 'r')
#ax.add_artist(plt.Circle((40,40), 20, color='k'))

nodes = []

nodes.append(start)
for i in range(NUMNODES):
    rand = random.random()*100, random.random()*100
#    while dist(rand, (40,40)) < 20:
#        rand = random.random()*100, random.random()*100
        
    cur_node = nodes[0]
    for p in nodes:
        if dist(p,rand) < dist(cur_node,rand):
            cur_node = p
    if dist(cur_node, goal) < EPSILON:
        plt.plot([cur_node[0], goal[0]], [cur_node[1], goal[1]], color = 'b')
        print(i)
        break
    newnode = step_from_to(cur_node,rand)
    nodes.append(newnode)
    
    plt.plot([cur_node[0], newnode[0]], [cur_node[1], newnode[1]], color = 'b')
    
#line = Line2D((50,50), (65,45))
#ax.add_line(line)

plt.show()

