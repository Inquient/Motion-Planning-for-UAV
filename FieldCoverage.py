import numpy as np
import matplotlib.pyplot as plt
from RRTLib import *
import matplotlib.lines
import matplotlib.patches as patches
import scipy.interpolate
import time


# Функция позволяет задать дробный рендж
def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


# Функция находин расстояние между двумя точками
def distance(x1, y1, x2, y2):
    return np.sqrt(np.power(x2 - x1, 2) +
                   np.power(y2 - y1, 2))


# Функция удаляет повторяющиеся элементы из списка
def remove_duplicates(l):
    n = []
    for i in l:
        if i not in n:
            n.append(i)
    return n


# Класс прямой - линии
class Line(object):
    def __init__(self, start_dot, end_dot):
        self.start_dot = start_dot
        self.end_dot = end_dot

    # Возвращает длинну прямой
    def get_line_length(self):
        return np.sqrt(np.power(self.end_dot[0] - self.start_dot[0], 2) +
                       np.power(self.end_dot[1] - self.start_dot[1], 2))

    # Возвращает прямую в виде списка из начальной и конечной точки
    def get_as_array(self):
        return [self.start_dot, self.end_dot]


# Класс двухмерного прямоугольника
class Rectangle(object):
    def __init__(self, left_bottom, right_bottom, right_top, left_top):
        self.left_bottom = left_bottom
        self.right_bottom = right_bottom
        self.right_top = right_top
        self.left_top = left_top

    # Возвращает квадрат в виде четырёх точек
    def get_as_array(self):
        return [self.left_bottom, self.right_bottom, self.right_top, self.left_top, self.left_bottom]

    # Находит короткую сторону в прямоугольнике и возвращает её в виде двух точек
    def get_short_side(self):
        a = Line(self.left_bottom, self.right_bottom)
        b = Line(self.left_bottom, self.left_top)

        a_len, b_len = a.get_line_length(), b.get_line_length()

        if a_len < b_len:
            return a.get_as_array(), b.get_as_array()
        else:
            return b.get_as_array(), a.get_as_array()

    # Строит в прямоугольнике предполагаемые линии облёта
    def build_field_coverage(self, camera_angle):
        short_side, long_side = self.get_short_side()
        camera_angle = camera_angle - camera_angle * 0.20
        start_dots = []
        end_dots = []
        if short_side[0][0] != short_side[1][0]:
            for i in frange(short_side[0][0] + camera_angle, short_side[0][1], camera_angle * 2):
                start_dots.append((i, short_side[0][1]))
                end_dots.append((i, long_side[0][1]))

        if short_side[0][1] != short_side[1][1]:
            for i in frange(short_side[0][1] + camera_angle, short_side[1][1], camera_angle * 2):
                start_dots.append((short_side[0][0], i))
                end_dots.append((long_side[1][0], i))

        for dot in start_dots:
            dot = (round(dot[0], 2), round(dot[1], 2))
            # plt.scatter(dot[0], dot[1], color = 'k')
        for dot in end_dots:
            dot = (round(dot[0], 2), round(dot[1], 2))
            # plt.scatter(dot[0], dot[1], color = 'k')

        result = []
        for i in range(0, len(start_dots)):
            result.append(Line(start_dots[i], end_dots[i]))
        # Возвращает список линий облёта
        return result

    # Рисует двухмерный прямоугольник
    def plot_rect(self):
        rect = self.get_as_array()
        for i in range(len(rect) - 1):
            plt.plot([rect[i][0], rect[i + 1][0]], [rect[i][1], rect[i + 1][1]], color='r')


# Класс двухмерного многоугольника
class Polygon(object):
    def __init__(self, dots):
        self.dots = dots
        self.dots.append(self.dots[0])

    # Возвращает многоугольник в виде списка его точек
    def get_as_array(self):
        return self.dots

    # Возвращает многоугольник в виде списка его линий - граней
    def get_as_lines(self):
        lines = []
        for i in range(0, len(self.dots) - 1):
            lines.append(Line(self.dots[i], self.dots[i + 1]))
        return lines

    # Возвращает площадь многоугольника
    def get_area(self):
        s1 = 0
        s2 = 0
        for i in range(0, len(dots) - 1):
            s1 += dots[i][0] * dots[i + 1][1]
            s2 += dots[i][1] * dots[i + 1][0]
        return (s1 - s2) / 2

    # Рисует двухмерный многоугольник - все его грани и точки
    def plot_poly(self):
        for i in range(len(self.dots) - 1):
            plt.plot([self.dots[i][0], self.dots[i + 1][0]], [self.dots[i][1], self.dots[i + 1][1]],
                     color='b')
            # plt.scatter(self.dots[i][0], self.dots[i][1], color='b')


# Функция находит точку пересечения двух прямых
# На вход принимает координаты начальной и конечной точки линии облёта, затем - грани многоугольника
# def lines_cross(x1, y1, x2, y2, a1, b1, a2, b2):
def lines_cross(f_line, p_edge):
    A1 = f_line.start_dot[1] - f_line.end_dot[1]
    B1 = f_line.end_dot[0] - f_line.start_dot[0]
    C1 = f_line.start_dot[0] * f_line.end_dot[1] - f_line.end_dot[0] * f_line.start_dot[1]
    A2 = p_edge.start_dot[1] - p_edge.end_dot[1]
    B2 = p_edge.end_dot[0] - p_edge.start_dot[0]
    C2 = p_edge.start_dot[0] * p_edge.end_dot[1] - p_edge.end_dot[0] * p_edge.start_dot[1]

    x, y = 0, 0

    if B1 * A2 - B2 * A1 and A1:
        y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
        x = (-C1 - B1 * y) / A1
    elif B1 * A2 - B2 * A1 and A2:
        y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
        x = (-C2 - B2 * y) / A2

    d = distance(p_edge.start_dot[0], p_edge.start_dot[1], p_edge.end_dot[0], p_edge.end_dot[1])
    d1 = distance(p_edge.start_dot[0], p_edge.start_dot[1], x, y)
    d2 = distance(p_edge.end_dot[0], p_edge.end_dot[1], x, y)

    # Если найденная точка лежит на грани многоугольника, то возвращаем её, иначе - 0
    if abs((d1 + d2) - d) <= 0.01:
        return (x, y)
    else:
        return 0


# Функция находит прямоугольник, описывающий данный многоугольник
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

    # Возвращает двухмерный прямоугольник
    return Rectangle((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y))


# Функция для интерполяции пути, подходит как для 2 так и для 3 измерений
def interpolate_path(arr):
    arr = np.array(arr)
    distance = np.cumsum(np.sqrt(np.sum(np.diff(arr, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    alpha = np.linspace(0, 1, 115)
    interpolator = scipy.interpolate.interp1d(distance, arr, kind='cubic', axis=0)
    return interpolator(alpha)


# Функция для сглаживания пути, подходит как для 2 так и для 3 измерений
def smooth_path(arr, rank):
    arr = np.array(arr)
    distance = np.cumsum(np.sqrt(np.sum(np.diff(arr, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    splines = [scipy.interpolate.UnivariateSpline(distance, cord, k=rank, s=.2) for cord in arr.T]
    alpha = np.linspace(0, 1, 75)
    return np.vstack(spl(alpha) for spl in splines).T


# Задаём двухмерное пространство
fig, ax = plt.subplots(1)

# Ещё один вариант многоугольника
# dots = [(1, 4), (4, 2), (4, 4), (16, 4), (24, 7), (24, 19), (17, 19), (8, 23), (6,13), (1,7)]

# Задаём многоугольник в виде множества точек и строим вокруг него прямоугольник
dots = [(150, 37.5), (300, 75), (300, 150), (337.5, 112.5), (412.5, 112.5), (512, 150), (512, 262.5),
        (412.5, 262.5), (412.5, 412.5), (300, 487.5), (262.5, 412.5), (262.5, 300), (187.5, 300),
        (150, 375), (75, 300), (112.5, 225), (37.5, 150), ]
rect = draw_rect_around(dots)
# rect.plot_rect()

# Задаём поле зрения камеры и строим линии облёта
camera_angle = 15.5  # В данном случае он равен 15.5, значить ширина одной полосы - 15.5 метров
flight_lines = rect.build_field_coverage(camera_angle)

# Строим многоугольник из точек и рисуем его
p = Polygon(dots)
poly_area = p.get_area()
print(poly_area)  # Получаем площадь многоугольника
p.plot_poly()

# Получаем многоугольник в виде граней
poly_lines = p.get_as_lines()

# Ищем точки пересечения каждой линии облёта с гранями многоугольника
# Именно это и будут наши маршрутные точки, добавляем их в отдельный список
route = []
for f_line in flight_lines:
    for p_edge in poly_lines:
        cross_dot = lines_cross(f_line, p_edge)
        # Если пересечение существует, добавляем его в список маршрутных точек и рисуем
        if cross_dot != 0:
            route.append([round(cross_dot[0], 2), round(cross_dot[1], 2)])
            plt.scatter(round(cross_dot[0], 2), round(cross_dot[1], 2), color='red')

# В сложных многоугольниках может существовать несколько пересечений на одной линии облёта
# Нам нужно всего два - первое на линии и последнее, поэтому:
# Находим лишние точки и заменяем их первой точкой на данной линии
for i in range(0, len(route) - 3):
    y = route[i][1]
    x = route[i][0]
    if route[i + 2][1] == y and route[i + 1][0] > x:
        route[i] = route[i + 2]
        route[i + 1] = route[i + 2]
    if route[i + 2][1] == y and route[i + 1][0] < x:
        route[i + 1] = route[i]
        route[i + 2] = route[i]
# Удаляем дублирующие точки на пересечениях с гранями
route = remove_duplicates(route)

# Выстраиваем точки в правильном порядке
for i in range(0, len(route), 4):
    temp = route[i]
    route[i] = route[i + 1]
    route[i + 1] = temp

# Поскольку в нашем распоряжении все маршрутные точки, можно строить сам путь

# 1) RRT-connect
# my_path = RRT_path(30.0, 10000, (0, 0, 0), (90, 90, 0))
# my_path.draw_multiple_paths_2d(route)

# Трёхмерной пространство для построения
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title("Dubins airplane trajectory")

# Построение пути Дубинса
dubins_path = Dubins_path(10, pi / 4, pi / 3)
for i in range(0, len(route) - 1):
    ax.scatter(route[i][0], route[i][1], 0, color='r')
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
    dubins_path.compute_dubins_path(start_node, end_node, ax)


# Строим линии предполагаемог облёта, рисуем их и считаем его общую длинну
# route_dots = []
# for i in range(0, len(route)-1,2):
#     route_dots.append(line((route[i][0],route[i][1]), (route[i+1][0],route[i+1][1])))
# print(route_dots)

# for r in route_dots:
#     rect = patches.Rectangle((r.end_dot[0], r.end_dot[1]-camera_angle), r.get_line_length(), camera_angle*2,
#                              facecolor='b', alpha=0.1)
#     ax.add_patch(rect)
#     plt.plot([r.start_dot[0], r.end_dot[0]], [r.start_dot[1], r.end_dot[1]], marker = 'o', color = 'red')
#
# s=0
# for l in route_dots:
#     s += l.get_line_length()
# print(s)

# Строит сетку для А* и сглаженный путь
# for i in frange (0,25,0.5):
#     plt.axvline(x=i, linewidth=0.5, color='k')
#     plt.axhline(y=i, linewidth=0.5, color='k')
#     for j in frange (0, 25,0.5):
#         plt.scatter(i,j,color = 'k', s =2)
#
# a_star_dots = [(1,3),(5,3),(15,3.5),(21,5.5),(1,5.5),(0.5,8),(25,8),
# (25,10),(2.5,10),(4.5,12.5),(25,12.5),(25,15),(5,15),
# (5.5,17.5),(25,17.5),(17.5,20),(6,20),(6.5,22),(10.5,22),]
#
# a_star_dots = smooth_path(a_star_dots,1)
# a_star_dots = smooth_path(a_star_dots,5)
# a_star_dots = interpolate_path(a_star_dots)
# a_star_dots = interpolate_path(a_star_dots)
#
# for i in range(0, len(a_star_dots)-1):
#     plt.scatter(a_star_dots[i][0],a_star_dots[i][1], color = 'k')
#     plt.plot([a_star_dots[i][0],a_star_dots[i+1][0]],[a_star_dots[i][1],a_star_dots[i+1][1]],
#              linewidth=2,color='k')

plt.show()
