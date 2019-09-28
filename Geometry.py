import numpy as np
from matplotlib import pyplot as plt

# Данный модуль содержит методы для работы с геометрическими фигурами:
# точками, прямыми, прямо- и многоугольниками


# Функция позволяет задать дробный рендж
def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

# Функция находин расстояние между двумя точками
def distance(x1, y1, x2, y2):
    return np.sqrt(np.power(x2 - x1, 2) +
                   np.power(y2 - y1, 2))

# Класс Точка, состоит из двух координат, целых или дробных
class Dot(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Возвращает точку в виде списка её координат
    def to_array(self):
        return [self.x, self.y]

# Класс Прямая, состоит из двух объектов класса точка - наальной и конечной
class Line(object):
    def __init__(self, start_dot, end_dot):
        self.start_dot = start_dot
        self.end_dot = end_dot

    # Возвращает длинну прямой
    def get_line_length(self):
        return np.sqrt(np.power(self.end_dot.x - self.start_dot.x, 2) +
                       np.power(self.end_dot.y - self.start_dot.y, 2))

    # Возвращает прямую в виде списка из списков с координатими начальной и конечной точек
    def get_as_array(self):
        return [self.start_dot.to_array(), self.end_dot.to_array()]

# Класс Прямоугольник, состоит из четырёх точек
class Rectangle(object):
    def __init__(self, left_bottom_dot, right_bottom_dot, right_top_dot, left_top_dot):
        self.left_bottom_dot = left_bottom_dot
        self.right_bottom_dot = right_bottom_dot
        self.right_top_dot = right_top_dot
        self.left_top_dot = left_top_dot

    # Возвращает список всех вершин прямоугольника в списка из списков с координатими их точек
    def get_as_array(self):
        return [self.left_bottom_dot.to_array(), self.right_bottom_dot.to_array(),
                self.right_top_dot.to_array(), self.left_top_dot.to_array(), self.left_bottom_dot.to_array()]

    # Находит короткую сторону в прямоугольнике и возвращает её вмести с длинной стороной
    def get_short_side(self):
        a = Line(self.left_bottom_dot, self.right_bottom_dot)
        b = Line(self.left_bottom_dot, self.left_top_dot)

        if a.get_line_length() < b.get_line_length():
            return a, b
        else:
            return b, a

    # Строит в прямоугольнике предполагаемые линии облёта
    def build_field_coverage(self, camera_angle):

        short_side, long_side = self.get_short_side()

        camera_angle = camera_angle - camera_angle * 0.20

        start_dots = []
        end_dots = []

        if short_side.start_dot.x != short_side.end_dot.x:
            for i in frange(short_side.start_dot.x  + camera_angle, short_side.end_dot.x, camera_angle):
                start_dots.append(Dot(i, short_side.start_dot.y))
                end_dots.append(Dot(i, long_side.start_dot.y))

        if short_side.start_dot.y != short_side.end_dot.y:
            for i in frange(short_side.start_dot.y + camera_angle, short_side.end_dot.y, camera_angle):
                start_dots.append(Dot(short_side.start_dot.x, i))
                end_dots.append(Dot(long_side.end_dot.x, i))

        for dot in start_dots:
            dot = (round(dot.x, 2), round(dot.y, 2))
            # plt.scatter(dot[0], dot[1], color = 'k')
        for dot in end_dots:
            dot = (round(dot.x, 2), round(dot.y, 2))
            # plt.scatter(dot[0], dot[1], color = 'k')

        result = []
        for i in range(0, len(start_dots)):
            result.append(Line(start_dots[i], end_dots[i]))
        # Возвращает список линий облёта
        return result

    # Рисует двухмерный прямоугольник
    # Для кооректной работы этого метода требуется инициализировать в matplotlib двухмернй пространство
    def plot_rect(self):
        rect = self.get_as_array()
        for i in range(len(rect) - 1):
            plt.plot([rect[i][0], rect[i + 1][0]], [rect[i][1], rect[i + 1][1]], color='r')

# Класс Многоугольник, продставлен в виде закольцованного массива точек
class Polygon(object):
    def __init__(self, dots):
        self.dots = dots # Массив точек
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
        for i in range(0, len(self.dots) - 1):
            s1 += self.dots[i].x * self.dots[i + 1].y
            s2 += self.dots[i].y * self.dots[i + 1].x
        return (s1 - s2) / 2

    # Рисует двухмерный многоугольник - все его грани и точки
    # Для кооректной работы этого метода требуется инициализировать в matplotlib двухмернй пространство
    def plot_poly(self, color):
        for i in range(len(self.dots) - 1):
            plt.plot([self.dots[i].x, self.dots[i + 1].x], [self.dots[i].y, self.dots[i + 1].y],
                     color=color)
            plt.scatter(self.dots[i].x, self.dots[i].y, color=color)
            plt.annotate('('+str(self.dots[i].x)+', '+ str(self.dots[i].y)+')',
                         xy=(self.dots[i].x, self.dots[i].y-15),
                         color = 'r', style = 'oblique', weight = 'bold')

# Функция рисует прямоугольник вокруг многоугольника
# Принимает объект типа Многоугольник
def draw_rect_around(polygon):
    max_x = polygon.dots[0].x
    max_y = polygon.dots[0].y
    min_x = polygon.dots[0].x
    min_y = polygon.dots[0].y

    for dot in polygon.dots:
        if dot.x > max_x:
            max_x = dot.x
        if dot.x < min_x:
            min_x = dot.x
        if dot.y > max_y:
            max_y = dot.y
        if dot.y < min_y:
            min_y = dot.y

    # Возвращает двухмерный прямоугольник
    return Rectangle(Dot(min_x, min_y), Dot(max_x, min_y), Dot(max_x, max_y), Dot(min_x, max_y))

# Функция находит точку пересечения двух прямых
def lines_cross(f_line, p_edge):
    A1 = f_line.start_dot.y - f_line.end_dot.y
    B1 = f_line.end_dot.x - f_line.start_dot.x
    C1 = f_line.start_dot.x * f_line.end_dot.y - f_line.end_dot.x * f_line.start_dot.y
    A2 = p_edge.start_dot.y - p_edge.end_dot.y
    B2 = p_edge.end_dot.x - p_edge.start_dot.x
    C2 = p_edge.start_dot.x * p_edge.end_dot.y - p_edge.end_dot.x * p_edge.start_dot.y

    x, y = 0, 0

    if B1 * A2 - B2 * A1 and A1:
        y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
        x = (-C1 - B1 * y) / A1
    elif B1 * A2 - B2 * A1 and A2:
        y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
        x = (-C2 - B2 * y) / A2

    d = p_edge.get_line_length()
    d1 = distance(p_edge.start_dot.x, p_edge.start_dot.y, x, y)
    d2 = distance(p_edge.end_dot.x, p_edge.end_dot.y, x, y)

    # Если найденная точка лежит на грани многоугольника, то возвращаем её, иначе - 0
    if abs((d1 + d2) - d) <= 0.01:
        return (x, y)
    else:
        return 0


# poly = Polygon([Dot(1, 4), Dot(5, 3), Dot(6, 8)])
# # poly.plot_poly('b')
# # rect = draw_rect_around(poly)
# # rect.plot_rect()
# # plt.show()