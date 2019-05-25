import numpy as np
import matplotlib.pyplot as plt

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

class line(object):
    def __init__(self, start_dot, end_dot):
        self.start_dot = start_dot
        self.end_dot = end_dot

    def get_line_length(self):
        return np.sqrt(np.power(self.end_dot[0] - self.start_dot[0], 2) +
                       np.power(self.end_dot[1] - self.start_dot[1], 2))

    def get_as_array(self):
        return [self.start_dot, self.end_dot]

class rectangle(object):
    def __init__(self, left_bottom, right_bottom, right_top, left_top):
        self.left_bottom = left_bottom
        self.right_bottom = right_bottom
        self.right_top = right_top
        self.left_top = left_top

    def get_as_array(self):
        return [self.left_bottom, self.right_bottom, self.right_top, self.left_top, self.left_bottom]

    def get_short_side(self):
        a = line(self.left_bottom, self.right_bottom)
        b = line(self.left_bottom, self.left_top)

        a_len, b_len = a.get_line_length(), b.get_line_length()

        if a_len < b_len:
            return a.get_as_array(), b.get_as_array()
        else:
            return b.get_as_array(), a.get_as_array()

    def build_field_coverage(self):
        short_side, long_side = self.get_short_side()
        print(short_side, long_side)

        route_dots = []
        camera_angle = 1.3
        if short_side[0][0] != short_side[1][0]:
            for i in frange(short_side[0][0]+camera_angle, short_side[0][1], camera_angle):
                route_dots.append((i, short_side[0][1]))
                route_dots.append((i, long_side[0][1]))
                # plt.scatter(i, short_side[0][1], color = 'k')

        if short_side[0][1] != short_side[1][1]:
            for i in frange(short_side[0][1]+camera_angle, short_side[1][1], camera_angle):
                route_dots.append((short_side[0][0], i))
                route_dots.append((long_side[1][0], i))
                # plt.scatter(short_side[0][0], i, color = 'k')

        print(route_dots)
        for dot in route_dots:
            plt.scatter(dot[0], dot[1], color = 'k')

    def plot_rect(self):
        rect = self.get_as_array()
        for i in range(len(rect) - 1):
            plt.plot([rect[i][0], rect[i + 1][0]], [rect[i][1], rect[i + 1][1]], color='r')

def lines_cross(x1,y1,x2,y2, a1,b1,a2,b2):
    A1 = y1 - y2
    B1 = x2 - x1
    C1 = x1*y2 - x2*y1
    A2 = b1 - b2
    B2 = a2 - a1
    C2 = a1*b2 - a2*b1

    x,y = 0,0

    if B1*A2 - B2*A1 and A1:
        y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
        x = (-C1 - B1 * y) / A1
    elif B1 * A2 - B2 * A1 and A2:
        y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
        x = (-C2 - B2 * y) / A2

    return(x,y)

def draw_rect_around(polygon_dots):
    max_x = polygon_dots[0][0]
    max_y = polygon_dots[0][1]
    min_x = polygon_dots[0][0]
    min_y = polygon_dots[0][1]

    for dot in polygon_dots:
        if dot[0] > max_x:
            max_x = dot[0]
        if dot[0] < min_x:
            min_x = dot[0]
        if dot[1] > max_y:
            max_y = dot[1]
        if dot[1] < min_y:
            min_y = dot[1]

    return rectangle((min_x,min_y),(max_x, min_y),(max_x, max_y),(min_x, max_y))


polygon = [(1, 4), (4, 2), (4, 4), (8, 6), (4, 7), (4, 9)]
rect = draw_rect_around(polygon)

polygon.append(polygon[0])
for i in range(len(polygon) - 1):
    plt.plot([polygon[i][0], polygon[i + 1][0]], [polygon[i][1], polygon[i + 1][1]], color ='b')
    plt.scatter(polygon[i][0], polygon[i][1])

rect.build_field_coverage()
rect.plot_rect()

z = lines_cross(1, 3.3, 8, 3.3, 1, 4, 4, 2)
print(z)
plt.scatter(z[0],z[1])

plt.show()