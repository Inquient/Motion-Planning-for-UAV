import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches as patches

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def distance(x1,y1,x2,y2):
    return np.sqrt(np.power(x2 - x1, 2) +
                    np.power(y2 - y1, 2))

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

    def build_field_coverage(self, camera_angle):
        short_side, long_side = self.get_short_side()
        camera_angle = camera_angle - camera_angle*0.20
        start_dots = []
        end_dots = []
        if short_side[0][0] != short_side[1][0]:
            for i in frange(short_side[0][0]+camera_angle, short_side[0][1], camera_angle*2):
                start_dots.append((i, short_side[0][1]))
                end_dots.append((i, long_side[0][1]))

        if short_side[0][1] != short_side[1][1]:
            for i in frange(short_side[0][1]+camera_angle, short_side[1][1], camera_angle*2):
                start_dots.append((short_side[0][0], i))
                end_dots.append((long_side[1][0], i))

        for dot in start_dots:
            dot = (round(dot[0],2),round(dot[1],2))
            # plt.scatter(dot[0], dot[1], color = 'k')
        for dot in end_dots:
            dot = (round(dot[0],2),round(dot[1],2))
            # plt.scatter(dot[0], dot[1], color = 'k')

        result = []
        for i in range(0,len(start_dots)):
            result.append(line(start_dots[i],end_dots[i]))

        return result

    def plot_rect(self):
        rect = self.get_as_array()
        for i in range(len(rect) - 1):
            plt.plot([rect[i][0], rect[i + 1][0]], [rect[i][1], rect[i + 1][1]], color='r')

class polygon(object):
    def __init__(self, dots):
        self.dots = dots
        self.dots.append(self.dots[0])

    def get_as_array(self):
        return self.dots

    def get_as_lines(self):
        lines = []
        for i in range(0, len(self.dots)-1):
            lines.append(line(self.dots[i], self.dots[i+1]))
        return lines

    def plot_poly(self):
        for i in range(len(self.dots) - 1):
            plt.plot([self.dots[i][0], self.dots[i + 1][0]], [self.dots[i][1], self.dots[i + 1][1]], color='b')
            # plt.scatter(self.dots[i][0], self.dots[i][1], color='b')

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

    d = distance(a1,b1,a2,b2)
    d1 = distance(a1,b1,x,y)
    d2 = distance(a2,b2,x,y)

    if abs((d1+d2)-d) <= 0.01:
        return (x,y)
    else:
        return 0

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

fig,ax = plt.subplots(1)

dots = [(1, 4), (4, 2), (4, 4), (16, 4), (24, 7), (24, 19), (17, 19), (8, 23), (6,13), (1,7)]
rect = draw_rect_around(dots)

# Точки на описывающем квадрате
camera_angle = 1.5
rect_dots = rect.build_field_coverage(camera_angle)
# for r in rect_dots:
#     print(r.get_as_array())

# Строим многоугольник из точек
p = polygon(dots)
p.plot_poly()

# Получаем многоугольник в виде граней
p = p.get_as_lines()
# for l in p:
#     print(l.get_as_array())

# Ищем точки пересечения каждой грани многоугольника с линиями облёта
route = []
for r in rect_dots:
    for l in p:
        z = lines_cross(r.start_dot[0], r.start_dot[1], r.end_dot[0], r.end_dot[1],
                        l.start_dot[0], l.start_dot[1], l.end_dot[0], l.end_dot[1])
        if z != 0:
            route.append([round(z[0],2),round(z[1],2)])
            plt.scatter(round(z[0],2),round(z[1],2),color='red')

# route_start = [dot for dot in route[1::2]]
# route_end = [dot for dot in route[::2]]
# print(route)

route_dots = []
for i in range(0, len(route)-1,2):

    route_dots.append(line((route[i][0],route[i][1]), (route[i+1][0],route[i+1][1])))
print(route_dots)

rect = patches.Rectangle((route_dots[0].start_dot[0], route_dots[0].start_dot[1]-camera_angle),
                         route_dots[0].get_line_length(), camera_angle*2,
                             facecolor='b', alpha=0.1)
ax.add_patch(rect)
plt.plot([route_dots[0].start_dot[0], route_dots[0].end_dot[0]],
         [route_dots[0].start_dot[1], route_dots[0].end_dot[1]], marker='o', color='red')
for r in route_dots[1:]:
    rect = patches.Rectangle((r.end_dot[0], r.end_dot[1]-camera_angle), r.get_line_length(), camera_angle*2,
                             facecolor='b', alpha=0.1)
    ax.add_patch(rect)
    plt.plot([r.start_dot[0], r.end_dot[0]], [r.start_dot[1], r.end_dot[1]], marker = 'o', color = 'red')


# rect.plot_rect()
plt.show()