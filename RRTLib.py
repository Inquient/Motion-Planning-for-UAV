#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import math
import numpy as np
from math import sqrt, cos, sin, atan2, tan
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate
from DubinsAirplaneFunctions import DubinsAirplanePath, MinTurnRadius_DubinsAirplane, ExtractDubinsAirplanePath

pi = np.pi

# class VehicleParameters(object):
#     """
#         Vehicle Parameters
#     """
#
#     def __init__(self, Vairspeed_0, Bank_max, Gamma_max):
#         self.Vairspeed_0 = Vairspeed_0
#         self.Bank_max = Bank_max
#         self.Gamma_max = Gamma_max

class Dubins_path:
    def __init__(self, Vairspeed_0, Bank_max, Gamma_max):
        self.Vairspeed_0 = Vairspeed_0
        self.Bank_max = Bank_max
        self.Gamma_max = Gamma_max

    def compute_dubins_path(self, route, ax):
        full_path = np.empty((0, 3))
        R_min = MinTurnRadius_DubinsAirplane(self.Vairspeed_0, self.Bank_max)

        for i in range(0, len(route) - 1):
            # ax.scatter(route[i][0], route[i][1], 0, color='r')
            if i % 4 == 0:
                start_node = np.array([route[i][0], route[i][1], 0, 0 * pi / 180, 1])
                end_node = np.array([route[i + 1][0], route[i + 1][1], 0, 0 * pi / 180, 1])
            elif i % 4 == 1:
                start_node = np.array([route[i][0], route[i][1], 0, 0 * pi / 180, 1])
                end_node = np.array([route[i + 1][0], route[i + 1][1], 0, 180 * pi / 180, 1])
            elif i % 4 == 2:
                start_node = np.array([route[i][0], route[i][1], 0, 180 * pi / 180, 1])
                end_node = np.array([route[i + 1][0], route[i + 1][1], 0, 180 * pi / 180, 1])
            else:
                start_node = np.array([route[i][0], route[i][1], 0, 180 * pi / 180, 1])
                end_node = np.array([route[i + 1][0], route[i + 1][1], 0, 0 * pi / 180, 1])

            DubinsAirplaneSolution1 = DubinsAirplanePath(start_node, end_node, R_min, self.Gamma_max)
            path_dubins_airplane1 = ExtractDubinsAirplanePath(DubinsAirplaneSolution1)
            path_dubins_airplane1 = path_dubins_airplane1.T
            full_path = np.vstack((full_path, path_dubins_airplane1))

        return full_path

class Potential_field_path:
    def __init__(self, route, obstacles):
        self.route = route
        self.obstacles = obstacles
        self.iterations = 0
        self.path_length = 0

    def nearest_dot(self, start, obstacles):
        dist = []
        for d in obstacles:
            dist.append(norm(start - d))
        return obstacles[np.argmin(dist)]

    def compute_potential_path(self):
        full_path = []
        for i in range(0, len(self.route)-1):
            full_path.extend(self.potential_path_to(self.route[i], self.route[i+1]))
        return  full_path

    def potential_path_to(self, start, goal):
        start = np.array(start)
        goal = np.array(goal)
        iter = 1000
        path = []
        while (norm(goal - start) > 1) and (iter > 0):
            attr_coeff = 2
            rep_coeff = 225

            f_attr = -(attr_coeff * (goal - start)) / norm(goal - start)
            f_rep = 0
            nearest_obstacle = self.nearest_dot(start, self.obstacles)
            if norm(nearest_obstacle - start) < 50:
                f_rep = rep_coeff * ((1 / (norm(nearest_obstacle - start)) - (1 / 10)) *
                                     (1 / math.pow(norm(nearest_obstacle - start), 2)) *
                                     ((nearest_obstacle - start) / norm(nearest_obstacle - start)))
            f = -(f_attr + f_rep)
            next = start + (2 * (f) / norm(f))
            self.path_length += norm(next - start)
            start += 2 * (f) / norm(f)
            path.append([start[0], start[1]])
            iter -= 1
            self.iterations += 1
        return path

class RRT_Dubins_path:
    def __init__(self, Vairspeed_0, Bank_max, Gamma_max, numnodes, eps):
        self.Vairspeed_0 = Vairspeed_0
        self.Bank_max = Bank_max
        self.Gamma_max = Gamma_max

        self.NUMNODES = numnodes
        self.EPSILON = eps
        self.path = []

    def dist_2d(self, p1, p2):
        return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

    def step_from_to_2d(self, p1, p2):
        if self.dist_2d(p1,p2) < self.EPSILON:
            return p2
        else:
            theta = atan2(p2[1]-p1[1],p2[0]-p1[0])
            p1 = p1[0] + self.EPSILON*cos(theta), p1[1] + self.EPSILON*sin(theta)
            return p1

    def draw_multiple_paths_2d(self, route, fig, ax):
        full = []

        for dot in route:
            ax.scatter(dot[0], dot[1], color = 'k')
        for i in range(0, len(route) - 1):
            self.start = route[i]
            self.goal = route[i + 1]
            if i%2 == 0:
                self.EPSILON = self.dist_2d(self.goal, self.start)
            self.rrt_connect_2d()
            full.append(self.path)

        full_path = np.empty((0, 3))
        # R_min = MinTurnRadius_DubinsAirplane(self.Vairspeed_0, self.Bank_max)
        R_min = 10
        self.Gamma_max = 1.5
        # start_node = np.array([self.start[0], self.start[1], 0, 0 * pi / 180, 1])
        # end_node = np.array([self.goal[0], self.goal[1], 0, 0 * pi / 180, 1])

        dir = 0
        for arr in full:
            for i in range(0, len(arr)-1):
                theta = atan2(arr[i+1][1] - arr[i][1], arr[i+1][0] - arr[i][0])

                # if dir % 4 == 0:
                #     start_node = np.array([arr[i][0], arr[i][1], 0, theta, 1])
                #     end_node = np.array([arr[i + 1][0], arr[i + 1][1], 0, theta, 1])
                # elif dir % 4 == 1:
                #     start_node = np.array([arr[i][0], arr[i][1], 0, theta, 1])
                #     end_node = np.array([arr[i + 1][0], arr[i + 1][1], 0, theta, 1])
                # elif dir % 4 == 2:
                #     start_node = np.array([arr[i][0], arr[i][1], 0, theta, 1])
                #     end_node = np.array([arr[i + 1][0], arr[i + 1][1], 0, theta, 1])
                # else:
                #     start_node = np.array([arr[i][0], arr[i][1], 0, theta, 1])
                #     end_node = np.array([arr[i + 1][0], arr[i + 1][1], 0, theta, 1])

                # delta_x = arr[i + 1][0] - arr[i][0]
                # if delta_x > 0:
                #     start_node = np.array([arr[i][0], arr[i][1], 0, 0 * pi / 180, 1])
                #     end_node = np.array([arr[i + 1][0], arr[i + 1][1], 0, 0 * pi / 180, 1])
                # else:
                #     start_node = np.array([arr[i][0], arr[i][1], 0, 180 * pi / 180, 1])
                #     end_node = np.array([arr[i + 1][0], arr[i + 1][1], 0, 180 * pi / 180, 1])

                if dir % 4 == 0:
                    start_node = np.array([arr[i][0], arr[i][1], 0, 0 * pi / 180, 1])
                    end_node = np.array([arr[i + 1][0], arr[i + 1][1], 0, 0 * pi / 180, 1])
                elif dir % 4 == 1:
                    start_node = np.array([arr[i][0], arr[i][1], 0, 0 * pi / 180, 1])
                    end_node = np.array([arr[i + 1][0], arr[i + 1][1], 0, 180 * pi / 180, 1])
                elif dir % 4 == 2:
                    start_node = np.array([arr[i][0], arr[i][1], 0, 180 * pi / 180, 1])
                    end_node = np.array([arr[i + 1][0], arr[i + 1][1], 0, 180 * pi / 180, 1])
                else:
                    start_node = np.array([arr[i][0], arr[i][1], 0, 180 * pi / 180, 1])
                    end_node = np.array([arr[i + 1][0], arr[i + 1][1], 0, 0 * pi / 180, 1])

                DubinsAirplaneSolution1 = DubinsAirplanePath(start_node, end_node, R_min, self.Gamma_max)
                path_dubins_airplane1 = ExtractDubinsAirplanePath(DubinsAirplaneSolution1)
                path_dubins_airplane1 = path_dubins_airplane1.T
                full_path = np.vstack((full_path, path_dubins_airplane1))
                # plt.plot([arr[i][0], arr[i+1][0]], [arr[i][1], arr[i+1][1]], color = 'r')
            dir += 1
        return full_path

        #plt.show()

    def rrt_connect_2d(self):
        start_nodes = []
        goal_nodes = []

        start_tree = {}
        goal_tree = {}
        current1 = 0
        current2 = 0

        start_nodes.append(self.start)
        goal_nodes.append(self.goal)

        lx, rx, by, uy = self.direction_check()

        flag = True

        for i in range(self.NUMNODES):
            rand1 = random.randrange(round(lx), round(rx)), random.randrange(round(by), round(uy))
            cur_node1 = start_nodes[0]
            for p in start_nodes:
                if self.dist_2d(p, rand1) < self.dist_2d(cur_node1, rand1):
                    cur_node1 = p
            newnode = self.step_from_to_2d(cur_node1, rand1)
            start_nodes.append(newnode)
            start_tree[newnode] = cur_node1

            rand2 = random.randrange(round(lx), round(rx)), random.randrange(round(by), round(uy))
            cur_node2 = goal_nodes[0]
            for p in goal_nodes:
                if self.dist_2d(p, rand2) < self.dist_2d(cur_node2, rand2):
                    cur_node2 = p
            newnode = self.step_from_to_2d(cur_node2, rand2)
            goal_nodes.append(newnode)
            goal_tree[newnode] = cur_node2

            for a in start_nodes:
                for b in goal_nodes:
                    if self.dist_2d(a, b) < self.EPSILON:
                        current1 = a
                        current2 = b
                        flag = False
                        break
                if not flag:
                    break
            if not flag:
                break

        start_path = [current1]
        while current1 != self.start:
            current1 = start_tree[current1]
            start_path.append(current1)

        goal_path = [current2]
        while current2 != self.goal:
            current2 = goal_tree[current2]
            goal_path.append(current2)

        start_path.reverse()
        self.path = start_path + goal_path

    def direction_check(self):
        zazor = 25
        if ((self.start[0] > self.goal[0]) and (self.start[1] == self.goal[1])):
            lx = self.goal[0] - zazor
            rx = self.start[0]
            by = self.start[1] - zazor
            uy = self.start[1] + zazor

        if ((self.start[0] > self.goal[0]) and (self.start[1] < self.goal[1])):
            lx = self.goal[0] - zazor
            rx = self.start[0]
            by = self.start[1]
            uy = self.goal[1] + zazor

        if ((self.start[0] > self.goal[0]) and (self.start[1] > self.goal[1])):
            lx = self.goal[0] - zazor
            rx = self.start[0]
            by = self.goal[1] - zazor
            uy = self.start[1]

        if ((self.start[0] < self.goal[0]) and (self.start[1] == self.goal[1])):
            lx = self.start[0]
            rx = self.goal[0] + zazor
            by = self.start[1] - zazor
            uy = self.start[1] + zazor

        if ((self.start[0] < self.goal[0]) and (self.start[1] > self.goal[1])):
            lx = self.start[0]
            rx = self.goal[0] + zazor
            by = self.goal[1] - zazor
            uy = self.start[1]

        if ((self.start[0] < self.goal[0]) and (self.start[1] < self.goal[1])):
            lx = self.start[0]
            rx = self.goal[0] + zazor
            by = self.start[1]
            uy = self.goal[1] + zazor

        if ((self.start[0] == self.goal[0]) and (self.start[1] > self.goal[1])):
            lx = self.start[0] - zazor
            rx = self.start[0] + zazor
            by = self.goal[1] - zazor
            uy = self.start[1]

        if ((self.start[0] == self.goal[0]) and (self.start[1] < self.goal[1])):
            lx = self.start[0] - zazor
            rx = self.start[0] + zazor
            by = self.start[1]
            uy = self.goal[1] + zazor

        return lx, rx, by, uy


class RRT_path:
    def __init__(self, eps, numnodes, start=(0,0), end=(50,50)):
        self.EPSILON = eps
        self.NUMNODES = numnodes
        self.start = start
        self.goal = end
        self.path = []
        self.path_length = 0
        
    def dist_2d(self, p1, p2):
        return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))
    
    def dist_3d(self, p1, p2):
        return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])+(p1[2]-p2[2])*(p1[2]-p2[2]))
    
    def step_from_to_2d(self, p1, p2):
        if self.dist_2d(p1,p2) < self.EPSILON:
            return p2
        else:
            theta = atan2(p2[1]-p1[1],p2[0]-p1[0])
            p1 = p1[0] + self.EPSILON*cos(theta), p1[1] + self.EPSILON*sin(theta)
            return p1
        
    def step_from_to_3d(self, p1, p2):
        if self.dist_3d(p1,p2) < self.EPSILON:
            return p2
        else:
            theta1 = atan2(p2[1]-p1[1],p2[0]-p1[0])
            theta2 = atan2(p2[2]-p1[2],p2[0]-p1[0])
            p1 = p1[0] + self.EPSILON*cos(theta1), p1[1] + self.EPSILON*sin(theta1), p1[2] + self.EPSILON*sin(theta2)
            return p1
        
    def rrt_2d_path(self):
        random_tree = {}
        current = 0
        nodes = []
        
        lx,rx,by,uy = self.direction_check()
        
        nodes.append(self.start)
        for i in range(self.NUMNODES):
            rand = random.randrange(round(lx),round(rx)), random.randrange(round(by),round(uy))
                
            cur_node = nodes[0]
            for p in nodes:
                if self.dist_2d(p,rand) < self.dist_2d(cur_node,rand):
                    cur_node = p
            if self.dist_2d(cur_node, self.goal) < self.EPSILON:
                random_tree[self.goal] = cur_node
                current = cur_node
                break
            newnode = self.step_from_to_2d(cur_node,rand)
            nodes.append(newnode)
            random_tree[newnode] = cur_node
        
        self.path = [current]
        while current != self.start:
            current = random_tree[current]
            self.path.append(current)
        self.path.reverse()
        self.path.append(self.goal)
        
    def rrt_connect_2d(self):
        start_nodes = []
        goal_nodes = []
        iterations = 0
        
        start_tree = {}
        goal_tree = {}
        current1 = 0
        current2 = 0
        
        start_nodes.append(self.start)
        goal_nodes.append(self.goal)
        
        lx,rx,by,uy = self.direction_check()
        
        flag = True
        
        for i in range(self.NUMNODES):
            rand1 = random.randrange(round(lx), round(rx)), random.randrange(round(by), round(uy))
            cur_node1 = start_nodes[0]
            for p in start_nodes:
                if self.dist_2d(p,rand1) < self.dist_2d(cur_node1,rand1):
                    cur_node1 = p
            newnode = self.step_from_to_2d(cur_node1,rand1)
            start_nodes.append(newnode)
            start_tree[newnode] = cur_node1
            
            rand2 = random.randrange(round(lx), round(rx)), random.randrange(round(by), round(uy))
            cur_node2 = goal_nodes[0]
            for p in goal_nodes:
                if self.dist_2d(p,rand2) < self.dist_2d(cur_node2,rand2):
                    cur_node2 = p
            newnode = self.step_from_to_2d(cur_node2,rand2)
            goal_nodes.append(newnode)
            goal_tree[newnode] = cur_node2

            for a in start_nodes:
                for b in goal_nodes:
                    if self.dist_2d(a, b) < self.EPSILON:
                        current1 = a
                        current2 = b
                        iterations += (i*2)
                        flag = False
                        break
                if not flag:
                    break
            if not flag:
                break
                
        start_path = [current1]
        while current1 != self.start:
                current1 = start_tree[current1]
                start_path.append(current1)
             
        goal_path = [current2]
        while current2 != self.goal:
                current2 = goal_tree[current2]
                goal_path.append(current2)
        
        start_path.reverse()
        self.path = start_path + goal_path
        return iterations

    def rrt_3d_path(self):
        random_tree = {}
        current = 0
        nodes = []
        
        lx,rx,by,uy = self.direction_check()
        
        nodes.append(self.start)
        for i in range(self.NUMNODES):
            rand = random.randrange(lx,rx), random.randrange(by,uy), random.randrange(0,100)
                
            cur_node = nodes[0]
            for p in nodes:
                if self.dist_3d(p,rand) < self.dist_3d(cur_node,rand):
                    cur_node = p
            if self.dist_3d(cur_node, self.goal) < self.EPSILON:
                random_tree[self.goal] = cur_node
                current = cur_node
                break
            newnode = self.step_from_to_3d(cur_node,rand)
            nodes.append(newnode)
            random_tree[newnode] = cur_node
                        
        self.path = [current]
        while current != self.start:
            current = random_tree[current]
            self.path.append(current)
        self.path.reverse()
        self.path.append(self.goal)
        
    def rrt_connect_3d(self):
        start_nodes = []
        goal_nodes = []
        
        start_tree = {}
        goal_tree = {}
        current1 = 0
        current2 = 0
        
        start_nodes.append(self.start)
        goal_nodes.append(self.goal)
        
        lx,rx,by,uy = self.direction_check()
        
        flag = True
        
        for i in range(self.NUMNODES):
            rand1 = random.randrange(lx,rx), random.randrange(by,uy), random.randrange(50,51)
            cur_node1 = start_nodes[0]
            for p in start_nodes:
                if self.dist_3d(p,rand1) < self.dist_3d(cur_node1,rand1):
                    cur_node1 = p
            newnode = self.step_from_to_3d(cur_node1,rand1)
            start_nodes.append(newnode)
            start_tree[newnode] = cur_node1
            
            rand2 = random.randrange(lx,rx), random.randrange(by,uy), random.randrange(50,51)
            cur_node2 = goal_nodes[0]
            for p in goal_nodes:
                if self.dist_3d(p,rand2) < self.dist_3d(cur_node2,rand2):
                    cur_node2 = p
            newnode = self.step_from_to_3d(cur_node2,rand2)
            goal_nodes.append(newnode)
            goal_tree[newnode] = cur_node2
                        
            for a in start_nodes:
                for b in goal_nodes:
                    if self.dist_3d(a, b) <self. EPSILON:
                        current1 = a
                        current2 = b
                        flag = False
                        break
                if not flag:
                    break
            if not flag:
                break
        
        start_path = [current1]
        while current1 != self.start:
                current1 = start_tree[current1]
                start_path.append(current1)
             
        goal_path = [current2]
        while current2 != self.goal:
                current2 = goal_tree[current2]
                goal_path.append(current2)
        
        start_path.reverse()
        self.path = start_path + goal_path
        
    def len_2d_path(self):
        for i in range(0, len(self.path)-1):
            self.path_length += self.dist_2d(self.path[i], self.path[i+1])
        return self.path_length

    def len_3d_path(self):
        for i in range(0, len(self.path)-1):
            self.path_length += self.dist_3d(self.path[i], self.path[i+1])
        return self.path_length
                    
    def show_2d_path(self):
        fig, ax = plt.subplots()
        plt.scatter(self.start[0], self.start[1], color = 'k')
        plt.scatter(self.goal[0], self.goal[1], color = 'r')
        for i in range(0, len(self.path)-1):
            plt.plot([self.path[i][0], self.path[i+1][0]], [self.path[i][1], self.path[i+1][1]], color = 'r')
            
    def show_3d_path(self):
        fig= plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.start[0], self.start[1], self.start[2], color = 'k')
        ax.scatter(self.goal[0], self.goal[1], self.goal[2], color = 'r')
        for i in range(0, len(self.path)-1):
            plt.plot([self.path[i][0], self.path[i+1][0]], [self.path[i][1], self.path[i+1][1]], [self.path[i][2], self.path[i+1][2]], color = 'r')

    def get_area(self, dots):
        s1 = 0
        s2 = 0
        for i in range(0, len(dots) - 1):
            s1 += dots[i][0] * dots[i + 1][1]
            s2 += dots[i][1] * dots[i + 1][0]
        return (s2 - s1) / 2

    def draw_multiple_paths_2d(self, dots, fig, ax):
        # fig, ax = plt.subplots()
        full = []
        len_full = 0
        iterations = []
        for dot in dots:
            ax.scatter(dot[0], dot[1], color = 'k')
        for i in range(0, len(dots)-1):
            self.start = dots[i]
            self.goal = dots[i+1]
            iterations.append(self.rrt_connect_2d())
            len_full += self.len_2d_path()
            full.append(self.path)

        covered_area = 0
        # return len_full, iterations
        for arr in full:
            # arr = self.smooth_path(arr)
            # arr = self.interpolate_path(arr)
            for i in range(0, len(arr)-1):
                plt.plot([arr[i][0], arr[i+1][0]], [arr[i][1], arr[i+1][1]], color = 'r')
                p = [[arr[i][0],arr[i][1]+15.5],[arr[i+1][0],arr[i+1][1]+15.5],
                     [arr[i+1][0],arr[i+1][1]-15.5],[arr[i][0],arr[i][1]-15.5],]
                rect = patches.Polygon(p, facecolor='b', alpha=0.1)
                covered_area += self.get_area(p)
                ax.add_patch(rect)

        print(covered_area)
        # plt.show()
        
    def draw_multiple_paths_3d(self, dots):
        fig= plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim(0,100)
        full = []
        for dot in dots:
            ax.scatter(dot[0], dot[1], dot[2], color = 'k')
        for i in range(0, len(dots)-1):
            self.start = dots[i]
            self.goal = dots[i+1]
            self.rrt_connect_3d()
            full.append(self.path)
        for arr in full:
            # arr = self.smooth_path(arr)
            # arr = self.interpolate_path(arr)
            for i in range(0, len(arr)-1):
                plt.plot([arr[i][0], arr[i+1][0]], [arr[i][1], arr[i+1][1]], [arr[i][2], arr[i+1][2]], color = 'r')
        plt.show()
        
    def direction_check(self):
        zazor = 25
        if((self.start[0] > self.goal[0]) and (self.start[1] == self.goal[1])):
            lx = self.goal[0]-zazor
            rx = self.start[0]
            by = self.start[1]-zazor 
            uy = self.start[1]+zazor
            
        if((self.start[0] > self.goal[0]) and (self.start[1] < self.goal[1])):
            lx = self.goal[0]-zazor
            rx = self.start[0]
            by = self.start[1]  
            uy = self.goal[1]+zazor
            
        if((self.start[0] > self.goal[0]) and (self.start[1] > self.goal[1])):
            lx = self.goal[0]-zazor
            rx = self.start[0]
            by = self.goal[1]-zazor 
            uy = self.start[1]      
            
        if((self.start[0] < self.goal[0]) and (self.start[1] == self.goal[1])):
            lx = self.start[0]
            rx = self.goal[0]+zazor
            by = self.start[1]-zazor 
            uy = self.start[1]+zazor
             
        if((self.start[0] < self.goal[0]) and (self.start[1] > self.goal[1])):
            lx = self.start[0]
            rx = self.goal[0]+zazor
            by = self.goal[1]-zazor
            uy = self.start[1]
            
        if((self.start[0] < self.goal[0]) and (self.start[1] < self.goal[1])):
            lx = self.start[0]
            rx = self.goal[0]+zazor
            by = self.start[1] 
            uy = self.goal[1]+zazor
            
        if((self.start[0] == self.goal[0]) and (self.start[1] > self.goal[1])):
            lx = self.start[0]-zazor
            rx = self.start[0]+zazor
            by = self.goal[1]-zazor  
            uy = self.start[1]
            
        if((self.start[0] == self.goal[0]) and (self.start[1] < self.goal[1])):
            lx = self.start[0]-zazor
            rx = self.start[0]+zazor
            by = self.start[1] 
            uy = self.goal[1]+zazor
            
        return lx,rx,by,uy
    
    def interpolate_path(self, arr):
        arr = np.array(arr)
        distance = np.cumsum( np.sqrt(np.sum( np.diff(arr, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
        alpha = np.linspace(0, 1, 75)
        interpolator =  scipy.interpolate.interp1d(distance, arr, kind='cubic', axis=0)
        return interpolator(alpha)
        
    def smooth_path(self, arr):
        arr = np.array(arr)
        distance = np.cumsum( np.sqrt(np.sum( np.diff(arr, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
        splines = [scipy.interpolate.UnivariateSpline(distance, cord, k=3, s=.2) for cord in arr.T]
        alpha = np.linspace(0, 1, 75)
        return np.vstack( spl(alpha) for spl in splines ).T



if __name__ == "__main__":
    my_path = RRT_path(10.0, 10000, (0,0,0), (90,90,0))

    dots = [(0,0,50), (0,90,50), (50,90,50), (50,0,50), (90,0,50), (90,90,50)]
    my_path.draw_multiple_paths_3d(dots)

    # dots = [(0,0), (0,90)]
    # my_path.draw_multiple_paths_2d(dots)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_title("Dubins airplane trajectory")
    #
    # d_path = Dubins_path(10, pi / 4, pi / 3)
    # d_path.compute_dubins_path(
    #     np.array([0,90,50,90*pi/180,15]), np.array([50,90,50,-120*pi/180,15]), ax)
    #
    # plt.show()


    