import numpy as np
import matplotlib.pyplot as plt
from RRTLib import *
import matplotlib.lines
import matplotlib.patches as patches
import scipy.interpolate
from Geometry import Dot,Line,Rectangle,Polygon,draw_rect_around,lines_cross
import time

# Функция позволяет задать дробный рендж
def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

# Функция удаляет повторяющиеся элементы из списка
def remove_duplicates(l):
    n = []
    for i in l:
        if i not in n:
            n.append(i)
    return n

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

# Функция позволяет нарисовать координатную сетку в заданном диапазоне
def draw_coordinate_net(range_start, range_end, step):
    for i in frange(range_start, range_end, step):
        plt.axvline(x=i, linewidth=0.5, color='k')
        plt.axhline(y=i, linewidth=0.5, color='k')
        for j in frange(range_start, range_end, step):
            plt.scatter(i, j, color='k', s=2)

# Возвращает маршрутные точки для облёта поля традиционным способом
# Принимает на вход список линий облёта и поле, для которого они были построены
def build_route_dots_for_field(flight_lines, field_polygon):
    # Представим многоугольник в виде списка его линий
    poly_lines = field_polygon.get_as_lines()

    route = []
    # Для каждой линии облёта найдём все её пересечения с линиями многоугольника
    for f_line in flight_lines:
        cross_cords = [[], []]
        for p_edge in poly_lines:
            # Находим точку пересечения
            cross_dot = lines_cross(f_line, p_edge)
            # Если точка пересечения существует, сохраняем её координаты в список
            # Отдельные списки для x и y координат
            if cross_dot != 0:
                cross_cords[0].append(round(cross_dot.x, 2))
                cross_cords[1].append(round(cross_dot.y, 2))
        # Маршрутных точек будет всего две, вне зависимости от количества пересечений
        # Так как это точки выхода с поля. Их коодинаты:
        route.append([max(cross_cords[0]), max(cross_cords[1])])
        route.append([min(cross_cords[0]), max(cross_cords[1])])

    return route


# Задаём двухмерное пространство
fig, ax = plt.subplots(1)

# Строим многоугольник в виде множества точек
p = Polygon([Dot(150, 37.5), Dot(300, 75), Dot(300, 150),
                Dot(337.5, 112.5), Dot(412.5, 112.5),
                Dot(512, 150), Dot(512, 262.5),
                Dot(412.5, 262.5), Dot(412.5, 412.5),
                Dot(300, 487.5), Dot(262.5, 412.5),
                Dot(262.5, 300), Dot(187.5, 300),
                Dot(150, 375), Dot(75, 300),
                Dot(112.5, 225), Dot(37.5, 150), ])

# Задаём поле зрения камеры и строим линии облёта
camera_angle = 31.0  # В данном случае он равен 15.5, значить ширина одной полосы - 15.5 метров
# Строим прямоугольник вокруг заданного многоугольника
rect = draw_rect_around(p)
flight_lines = rect.build_field_coverage(camera_angle)
route = build_route_dots_for_field(flight_lines, p)

# Отрисовываем заданный многоугольник
p.plot_poly('b', True)

for line in flight_lines:
    line.plot_line()

for dot in route:
    plt.scatter(dot[0], dot[1], color='red')

# Выстраиваем точки в правильном порядке
for i in range(0, len(route), 4):
    temp = route[i]
    route[i] = route[i + 1]
    route[i + 1] = temp

# draw_coordinate_net(0, 550, 20)

# Запретные зоны
# p1 = Polygon([(150,100),(150,200),(220,210),(220,170),(180,120),])
# p2 = Polygon([(262.5,320),(262.5,410),(325,350),])
# p1.plot_poly('r')
# p2.plot_poly('r')
# rect1 = patches.Polygon(p1.get_as_array(), facecolor='r', alpha=0.3)
# rect2 = patches.Polygon(p2.get_as_array(), facecolor='r', alpha=0.3)
# ax.add_patch(rect1)
# ax.add_patch(rect2)

# 0) Строим линии предполагаемог облёта, рисуем их и считаем его общую длинну
# route_lines = []
# for i in range(0, len(route)-1,2):
#     route_lines.append(Line((route[i][0], route[i][1]), (route[i + 1][0], route[i + 1][1])))
# print(route_lines)
#
# covered_area = 0
# for r in route_lines:
#     rect = patches.Rectangle((r.end_dot[0], r.end_dot[1]-camera_angle/2), r.get_line_length(), camera_angle,
#                              facecolor='b', alpha=0.1)
#     covered_area += r.get_line_length() * (camera_angle - camera_angle*0.2)
#     ax.add_patch(rect)
#     plt.plot([r.start_dot[0], r.end_dot[0]], [r.start_dot[1], r.end_dot[1]], marker = 'o', color = 'red')
#
# s=0
# for l in route_lines:
#     s += l.get_line_length()
# print(s)
# print(covered_area)

# Поскольку в нашем распоряжении все маршрутные точки, можно строить сам путь
# 1) RRT-connect
# times = []
# lengths = []
# iterations = []
# experiments = 100

# my_path = RRT_path(30.0, 10000, (0, 0, 0), (90, 90, 0))
# my_path.draw_multiple_paths_2d(route, fig, ax)

# for x in range(0, experiments):
#     t = time.time()
#     cur_len, cur_iter =  my_path.draw_multiple_paths_2d(route)
#     lengths.append(cur_len)
#     iterations.append(sum(cur_iter))
#     times.append(time.time()-t)
#
# print("Время, для нахождения пути")
# print("Среднее = ", np.average(times))
# print("Min = ", np.min(times))
# print("Max = ", np.max(times))
# print("SD = ", np.std(times))
#
# print("Длинна итогового пути")
# print("Среднее = ", np.average(lengths))
# print("Min = ", np.min(lengths))
# print("Max = ", np.max(lengths))
# print("SD = ", np.std(lengths))
#
# print("Количество узлов, для нахождения пути")
# print("Среднее = ", np.average(iterations))
# print("Min = ", np.min(iterations))
# print("Max = ", np.max(iterations))
# print("SD = ", np.std(iterations))

# 2) Алгоритм Дубинса
# Трёхмерной пространство для построения
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_title("Dubins airplane trajectory")

# p.plot_poly_in_3d(ax, [-1,-1])
# times = []
# t = time.time()
# dubins_path = Dubins_path(10, pi / 4, pi / 3)
# full_path = dubins_path.compute_dubins_path(route, ax)
# times.append(time.time()-t)
# ax.plot(full_path[:, 0], full_path[:, 1], 'k')
# length = 0
# for i in range(0,len(full_path)-1):
#     length += distance(full_path[i][0], full_path[i][1], full_path[i+1][0], full_path[i+1][1])
#
# print("Длинна итогового пути")
# print(length)
#
# print("Время, для нахождения пути")
# print("Среднее = ", np.average(times))
# print("Min = ", np.min(times))
# print("Max = ", np.max(times))
# print("SD = ", np.std(times))

# 3) Строит сетку для А* и сглаженный путь
# a_star_dots = route
# a_star_dots = []
# for i in range(0, len(route)-1):
#     a_star_dots.append(route[i])
#     if i%3 == 0:
#         a_star_dots.append()

# t = time.time()
# a_star_dots = smooth_path(a_star_dots,1)
# a_star_dots = smooth_path(a_star_dots,2)
# a_star_dots = interpolate_path(a_star_dots)
# # a_star_dots = interpolate_path(a_star_dots)
# t = time.time() - t
# print("Время, для нахождения пути = ", t)
#
# a_star_len = 0
# covered_area = 0
# for i in range(0, len(a_star_dots)-1):
#     plt.scatter(a_star_dots[i][0],a_star_dots[i][1], color = 'k')
#     a_star_len += distance(a_star_dots[i][0], a_star_dots[i][1],a_star_dots[i+1][0], a_star_dots[i+1][1])
#     plt.plot([a_star_dots[i][0],a_star_dots[i+1][0]],[a_star_dots[i][1],a_star_dots[i+1][1]],
#              linewidth=2,color='k')
#     p = [[a_star_dots[i][0], a_star_dots[i][1] + 15.5], [a_star_dots[i + 1][0], a_star_dots[i + 1][1] + 15.5],
#          [a_star_dots[i + 1][0], a_star_dots[i + 1][1] - 15.5], [a_star_dots[i][0], a_star_dots[i][1] - 15.5], ]
#     rect = patches.Polygon(p, facecolor='b', alpha=0.1)
#     covered_area += get_area(p)
#     ax.add_patch(rect)
#
# print("Длинна итогового пути",a_star_len)
# print('Площадь покрытия = ', covered_area)
#
# 4) Потенцыальное поле
# obstacles = np.array(((272.5,73.5),(274.5,73.5),(270.5,73.5),(268.5,73.5),(276.5,73.5),
#                       (286,80),(288,80),(290,80),(292,80),(294,80),
#                       (80,107),(82,107),(78,107),
#                       (445,132),(440,132),(435,132),(430,132),
#                       (50,148),
#                       (512,171),
#                       (88,200),
#                       (512,221),
#                       (101,248),
#                       (457,275),
#                       (84,299),
#                       (411,322),
#                       (124,350),
#                       (411,372),
#                       (261,397),
#                       (396,423),
#                       (280,448),
#                       (321,474),))
# poten_path = Potential_field_path(route, obstacles)
# t = time.time()
# full_path = poten_path.compute_potential_path()
# t = time.time()-t
#
# print("Время, для нахождения пути = ", t)
# print("Количество итераций, для нахождения пути = ", poten_path.iterations)
# print("Длинна итогового пути", poten_path.path_length)
#
# for dot in obstacles:
#     plt.scatter(dot[0], dot[1], color = 'r')
#
# covered_area = 0
# for dot in full_path:
#     plt.scatter(dot[0],dot[1], color = 'k', s=3)
#     # rect = patches.Rectangle((dot[0]-1,dot[1]-camera_angle/2),2,camera_angle,facecolor='b', alpha=0.1)
#     covered_area += camera_angle * 2
#     # ax.add_patch(rect)
# print('Площадь покрытия = ', covered_area)

# 5) RRT-Dubins
# rrt_dubins_path = RRT_Dubins_path(3, pi / 4, pi / 3, 10000, 40.0)
# full_path = rrt_dubins_path.draw_multiple_paths_2d(route, fig, ax)
# ax.plot(full_path[:, 0], full_path[:, 1], 'k')

plt.show()
