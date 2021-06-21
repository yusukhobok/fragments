import uuid

import numpy as np

import unGlobal
from unGlobal import HodError


class AmplitudeMapElement:
    """Класс элемента на амплитудных картах"""
    def __init__(self, caption, slice_num, kind):
        self.caption = caption
        self.slice_num = slice_num
        self.color_by_slice = True
        self.color = 'k'
        self.visible = True
        self.visible_in_all_slices = False
        self.kind = kind
        self.id = uuid.uuid1().int

    def set_caption(self, caption):
        self.caption = caption

    def set_slice_num(self, slice_num):
        self.slice_num = slice_num

    def set_color(self, by_slice, color=None):
        if not by_slice:
            self.color_by_slice = False
            try:
                self.color = color
            except ValueError:
                print("Такого цвета не существует")
                print(color)
        else:
            self.color_by_slice = True
            self.color = None

    def set_visible(self, v):
        self.visible = v

    def set_visible_in_all_slices(self, v):
        self.visible_in_all_slices = v

    def set_color_by_slice(self, v):
        self.color_by_slice = v


class AmplitudeMapPoint(AmplitudeMapElement):
    """Класс точки на амплитудных картах"""
    def __init__(self, X, Y, caption, slice_num):
        AmplitudeMapElement.__init__(self, caption, slice_num, "точка")
        self.X = X
        self.Y = Y

    def set_point(self, X, Y):
        self.X = X
        self.Y = Y

    def set_X(self, X):
        self.X = X

    def set_Y(self, Y):
        self.Y = Y


class AmplitudeMapRectangle(AmplitudeMapElement):
    """Класс прямоугольника на амплитудных картах"""
    def __init__(self, X0, Y0, X1, Y1, caption, slice_num):
        AmplitudeMapElement.__init__(self, caption, slice_num, "прямоугольник")
        self.X0 = X0
        self.Y0 = Y0
        self.X1 = X1
        self.Y1 = Y1

    def set_point0(self, X0, Y0):
        self.X0 = X0
        self.Y0 = Y0

    def set_X0(self, X0):
        self.X0 = X0

    def set_Y0(self, Y0):
        self.Y0 = Y0

    def set_point1(self, X1, Y1):
        self.X1 = X1
        self.Y1 = Y1

    def set_X1(self, X1):
        self.X1 = X1

    def set_Y1(self, Y1):
        self.Y1 = Y1


class AmplitudeMapText(AmplitudeMapElement):
    """Класс текста"""
    def __init__(self, X, Y, height, caption, slice_num):
        AmplitudeMapElement.__init__(self, caption, slice_num, "текст")
        self.X = X
        self.Y = Y
        self.height = height

    def set_point(self, X, Y):
        self.X = X
        self.Y = Y

    def set_X(self, X):
        self.X = X

    def set_Y(self, Y):
        self.Y = Y

    def set_height(self, height):
        self.height = height


class AmplitudeMapPolyline(AmplitudeMapElement):
    """Класс полилинии на амплитудных картах"""

    def __init__(self, vertexes, closed, caption, slice_num):
        AmplitudeMapElement.__init__(self, caption, slice_num, "ломаная")
        self.vertexes = np.array(vertexes)
        self.vertexes = self.vertexes[np.unique(self.vertexes[:, 0], return_index=True)[1]]  # удаление повторяющихся значений L
        self.closed = closed
        self.currVertexNum = 0

    def change_closed(self, closed):
        self.closed = closed

    def set_vertexes(self, vertexes):
        self.vertexes = vertexes
        self.vertexes = self.vertexes[np.unique(self.vertexes[:, 0], return_index=True)[1]]  # удаление повторяющихся значений L

    def add_vertex(self, vertex):
        vertex = np.array(vertex)
        OldVertexes = self.vertexes
        self.vertexes = np.append(self.vertexes, [vertex], axis=0)
        if len(self.vertexes) < 2:
            self.vertexes = OldVertexes
            raise HodError("Ломаная должна включать не менее двух разных точек")
        self.currVertexNum = len(self.vertexes)-1

    def insert_vertex(self, num, vertex):
        vertex = np.array(vertex)
        if num < 0 or num >= len(self.vertexes): return
        OldVertexes = self.vertexes
        self.vertexes = np.insert(self.vertexes, num, vertex, axis=0)
        if len(self.vertexes) < 2:
            self.vertexes = OldVertexes
            raise HodError("Ломаная должна включать не менее двух разных точек")

    def del_vertex(self, num):
        if len(self.vertexes) <= 2: return
        if num<0 or num >= len(self.vertexes): return
        OldVertexes = self.vertexes
        self.vertexes = np.delete(self.vertexes, num, axis=0)
        if len(self.vertexes) < 2:
            self.vertexes = OldVertexes
            raise HodError("Ломаная должна включать не менее двух разных точек")
        self.currVertexNum = max(0,num-1)

    def edit_vertex(self, num, vertex):
        vertex = np.array(vertex)
        if num<0 or num >= len(self.vertexes): return
        OldVertexes = self.vertexes
        self.vertexes[num] = vertex
        if len(self.vertexes) < 2:
            self.vertexes = OldVertexes
            raise HodError("Ломаная должна включать не менее двух разных точек")

    def find_nearest_vertex(self, x, y):
        X = self.vertexes[:,0]
        Y = self.vertexes[:,1]
        Dist = np.sqrt((X-x)**2 + (Y-y)**2)
        return np.argmin(Dist)

    def move(self, X, Y):
        if self.vertexes == None: return
        dX = X - self.vertexes[0,0]
        dY = Y - self.vertexes[0,1]
        self.move_delta(dX, dY)

    def move_delta(self, dX, dY):
        if self.vertexes == None: return
        self.vertexes[:,0] = self.vertexes[:,0] + dX
        self.vertexes[:,1] = self.vertexes[:,1] + dY
