import numpy as np
from matplotlib import pyplot as plt

# Функция позволяет задать дробный рендж
def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


class Dot(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_array(self):
        return [self.x, self.y]


# Прямая состоит из двух объектов класса точка
class Line(object):
    def __init__(self, start_dot, end_dot):
        self.start_dot = start_dot
        self.end_dot = end_dot

    # Возвращает длинну прямой
    def get_line_length(self):
        return np.sqrt(np.power(self.end_dot.x - self.start_dot.x, 2) +
                       np.power(self.end_dot.y - self.start_dot.y, 2))

    # Возвращает прямую в виде списка из начальной и конечной точек
    def get_as_array(self):
        return [self.start_dot.to_array(), self.end_dot.to_array()]


class Rectangle(object):
    def __init__(self, left_bottom_dot, right_bottom_dot, right_top_dot, left_top_dot):
        self.left_bottom_dot = left_bottom_dot
        self.right_bottom_dot = right_bottom_dot
        self.right_top_dot = right_top_dot
        self.left_top_dot = left_top_dot

    # Возвращает список всех вершин прямокгольника в виде координат их точек
    def get_as_array(self):
        return [self.left_bottom_dot.to_array(), self.right_bottom_dot.to_array(),
                self.right_top_dot.to_array(), self.left_top_dot.to_array(), self.left_bottom_dot.to_array()]

    # Находит короткую сторону в прямоугольнике и возвращает её вмести с длинной
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
                start_dots.append((i, short_side.start_dot.y))
                end_dots.append((i, long_side.start_dot.y))

        if short_side.start_dot.y != short_side.end_dot.y:
            for i in frange(short_side.start_dot.y + camera_angle, short_side.end_dot.y, camera_angle):
                start_dots.append((short_side.start_dot.x, i))
                end_dots.append((long_side.end_dot.x, i))

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


fig, ax = plt.subplots(1)
rect = Rectangle(Dot(1, 1), Dot(1, 5), Dot(4, 5), Dot(4, 1))
print(rect.build_field_coverage(0.1))
rect.plot_rect()
plt.show()
