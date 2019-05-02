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
NUMNODES = 10000

start = (0,0)
goal = (100, 100)

random_tree = {}
current = 0

fig, ax = plt.subplots()
plt.scatter(start[0], start[1], color = 'k')
plt.scatter(goal[0], goal[1], color = 'r')

nodes = []

nodes.append(start)
for i in range(NUMNODES):
    rand = random.random()*100, random.random()*100
        
    cur_node = nodes[0]
    for p in nodes:
        if dist(p,rand) < dist(cur_node,rand):
            cur_node = p
    if dist(cur_node, goal) < EPSILON:
        plt.plot([cur_node[0], goal[0]], [cur_node[1], goal[1]], color = 'b')
        random_tree[goal] = cur_node
        current = cur_node
        print(i)
        break
    newnode = step_from_to(cur_node,rand)
    nodes.append(newnode)
    random_tree[newnode] = cur_node
    
    plt.plot([cur_node[0], newnode[0]], [cur_node[1], newnode[1]], color = 'b')
    

#plt.show()
#print(random_tree)
print(len(random_tree))

path = [current]
while current != start:
        current = random_tree[current]
        path.append(current)

#print(path)
print(len(path))

path_length = 0
for i in range(0, len(path)-1):
    path_length += dist(path[i], path[i+1])
    plt.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]], color = 'r')

print(path_length)
plt.show()
