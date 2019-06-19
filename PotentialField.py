#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from numpy.linalg import norm
from numpy import gradient
import matplotlib.pyplot as plt 

def nearest_dot(start, obstacles):
    dist = []
    for d in obstacles:
        dist.append(norm(start - d))
    return obstacles[np.argmin(dist)]

def compute_potential_field(start, goal):
    iter = 1000
    path = []
    while (norm(goal - start) > 1) and (iter > 0):
        attr_coeff = 2
        rep_coeff = 5

        f_attr = -(attr_coeff * (goal - start)) / norm(goal - start)
        f_rep = 0
        nearest_obstacle = nearest_dot(start, obstacles)
        if norm(nearest_obstacle - start) < 10:
            f_rep = rep_coeff * ((1 / (norm(nearest_obstacle - start)) - (1 / 10)) *
                                 (1 / math.pow(norm(nearest_obstacle - start), 2)) *
                                 ((nearest_obstacle - start) / norm(nearest_obstacle - start)))
        f = -(f_attr + f_rep)
        start += 0.5 * (f) / norm(f)
        path.append([start[0], start[1]])
        iter -= 1
    return path

start = np.array((10.0,10.5))
goal = np.array((30.0,30.0))
obstacles = np.array(((15.0,15.0), (15.0,18.5), (22.0,24.0),(22.0,23.0),(22.0,22.0),
                      (28.0,28.0),(29.0,27.0),(26.0, 30.0),))
# obstacles = np.array(((20.0,20.0),(19.0,20.0),(18.0,20.0),(17.0,20.0),(16.0,20.0),(15.0,20.0),
#                       (20.0, 19.0), (20.0, 18.0), (20.0, 17.0), (20.0, 16.0), (20.0, 15.0),))

fig, ax = plt.subplots()
plt.scatter(goal[0], goal[1], color='g')
for dot in obstacles:
    plt.scatter(dot[0],dot[1], color = 'r')

path = compute_potential_field(start, goal)
for dot in path:
    plt.scatter(dot[0], dot[1], color='k')

plt.show()
