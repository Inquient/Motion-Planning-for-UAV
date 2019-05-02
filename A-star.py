#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from queue import Queue
import time


graph = { "a" : ["c"],
          "b" : ["c", "e"],
          "c" : ["a", "b", "d", "e"],
          "d" : ["c"],
          "e" : ["c", "b", "f"],
          "f" : ["e"]
        }

start = 'a'
goal = "f"

s = time.time()

frontier = Queue()
frontier.put(start)
came_from = {}
came_from[start] = None

while not frontier.empty():
   current = frontier.get()
   if current == goal:
       break
   for next in graph[current]:
      if next not in came_from:
         frontier.put(next)
         came_from[next] = current
         
print(came_from)

current = goal
path = [current]
while current != start: 
   current = came_from[current]
   path.append(current)
   
print(time.time() - s)
path.reverse()
   
print(path)