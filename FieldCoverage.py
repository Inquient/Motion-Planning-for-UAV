import numpy as np
import matplotlib.pyplot as plt

def expand_field(initial_dots):
    max_x = initial_dots[0][0]
    max_y = initial_dots[0][1]
    min_x = initial_dots[0][0]
    min_y = initial_dots[0][1]

    for dot in initial_dots:
        if dot[0] > max_x:
            max_x = dot[0]
        if dot[0] < min_x:
            min_x = dot[0]
        if dot[1] > max_y:
            max_y = dot[1]
        if dot[1] < min_y:
            min_y = dot[1]

    return [(min_x,min_y),(max_x, min_y),(max_x, max_y),(min_x, max_y),]


dots = [(1,4),(4,2),(4,4),(8,6),(4,7),]
rectangle = expand_field(dots)

dots.append(dots[0])
for i in range(len(dots)-1):
    plt.plot([dots[i][0], dots[i+1][0]],[dots[i][1], dots[i+1][1]], color = 'b')
    plt.scatter(dots[i][0], dots[i][1])

rectangle.append(rectangle[0])
for i in range(len(rectangle)-1):
    plt.plot([rectangle[i][0], rectangle[i + 1][0]], [rectangle[i][1], rectangle[i + 1][1]], color='r')

plt.show()