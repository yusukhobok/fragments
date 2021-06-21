import csv
import json
import copy
import pickle
import struct
from os.path import dirname, exists, basename

from PyQt5 import QtGui
import numpy as np
import xarray as xr

from amplitudemap.element import AmplitudeMapPoint, AmplitudeMapRectangle, AmplitudeMapText, AmplitudeMapPolyline
from unGlobal import go
from unProject import Project
from unSurvey import Survey
from PyQt5.QtCore import QObject
from PyQt5.QtCore import pyqtSignal


class Slices:
    """Класс амплитудных карт (георадарных срезов)"""
    def __init__(self):
        self.survey_file_name = None  # имя исходного файла проекта
        self.slices_data = None  # данные об амплитудных картах
        self.elements = []  # элементы на амплитудных картах
        self.useful_area_points = []  # полезная область на амплитудных картах
        self.selected_element = None  # выделенный элемент
        self.file_name = None  # имя файла амплитудных карт
        self.events = AmplitudeMapEvents()
        self.is_changed = False

    def save_object(self, f, obj):
        data = pickle.dumps(obj)
        n = len(data)
        f.write(struct.pack("i", n))
        f.write(data)

    def save(self, survey_file_name):
        FILE_FORMAT_VERSION = 2
        base = basename(survey_file_name)
        self.file_name = dirname(survey_file_name) + "//амплитудные карты_" + base + "_.ampl"
        with open(self.file_name, "wb") as f:
            self.save_object(f, self.slices_data)
            self.save_object(f, self.elements)
            self.save_object(f, FILE_FORMAT_VERSION)
            self.save_object(f, self.useful_area_points)
        self.is_changed = False

    def load_object(self, f):
        n, = struct.unpack("i", f.read(struct.calcsize("i")))
        data = f.read(n)
        return pickle.loads(data)

    def load(self, survey_file_name):
        base = basename(survey_file_name)
        self.file_name = dirname(survey_file_name) + "//амплитудные карты_" + base + "_.ampl"
        if exists(self.file_name):
            with open(self.file_name, "rb") as f:
                self.slices_data = self.load_object(f)
                self.elements = self.load_object(f)
                try:
                    version = self.load_object(f)
                except struct.error:
                    version = 1
                if version >= 2:
                    self.useful_area_points = self.load_object(f)
                else:
                    self.clear_useful_area()
            self.selected_element = None

            self.survey_file_name = survey_file_name

            go.opts.amplitudemap.rough_detail = self.slices_data.attrs['rough_detail']
            go.opts.amplitudemap.fine_detail = self.slices_data.attrs['fine_detail']
            go.opts.amplitudemap.coef_detality = self.slices_data.attrs['coef_detality']
            try:
                go.opts.amplitudemap.points_on_axis_count = self.slices_data.attrs['points_on_axis_count']
            except KeyError:
                self.slices_data.attrs['points_on_axis_count'] = 1000
                go.opts.amplitudemap.points_on_axis_count = 1000
            try:
                go.opts.amplitudemap.is_amplitudemap_for_linear_project = self.slices_data.attrs['is_amplitudemap_for_linear_project']
            except KeyError:
                self.slices_data.attrs['is_amplitudemap_for_linear_project'] = False
                go.opts.amplitudemap.is_amplitudemap_for_linear_project = False
            go.opts.amplitudemap.mode = self.slices_data.attrs['mode']
            go.opts.amplitudemap.show_trajectories = self.slices_data.attrs['show_trajectories']
            go.opts.amplitudemap.color_map = self.slices_data.attrs['color_map']
            go.opts.amplitudemap.contrast = self.slices_data.attrs['contrast']
            go.opts.amplitudemap.min_hist_value = self.slices_data.attrs['min_hist_value']
            go.opts.amplitudemap.max_hist_value = self.slices_data.attrs['max_hist_value']
            go.opts.amplitudemap.is_absolute_mode_hist = self.slices_data.attrs['is_absolute_mode_hist']
            go.opts.amplitudemap.is_consider_zero = self.slices_data.attrs['is_consider_zero']

            self.is_changed = False
            return True
        else:
            return False

    def refresh_slices(self):
        if self.slices_data is None:
            return
        self.generate_slices(self.survey_file_name, self.slices_data.attrs['time_delta'])

    def _generate_data_for_one_project(self, file_name, delta_time, delta_distance, time_max, time_delta, kind_data):
        project = Project()
        project.open(file_name, is_message=False)

        if (delta_time is not None) and (np.abs(delta_time - project.radargram.delta_time) > 0.001):
            return False, "Шаг между временными отсчетами должен быть одинаковым для всех файлов"
        if delta_time is None:
            delta_time = project.radargram.delta_time
        if project.radargram.time_max > time_max:
            time_max = project.radargram.time_max
        if (delta_distance is None) or (delta_distance > project.radargram.delta_distance):
            delta_distance = project.radargram.delta_distance

        if time_delta < delta_time:
            time_delta = delta_time
        if time_delta > time_max:
            time_delta = time_max
        if kind_data == 0:
            data = project.radargram.data_trace
        elif kind_data == 1:
            data = project.radargram.data_trace * project.radargram.gain_coef_array
        else:
            data = project.radargram.data_attribute
            if data is None:
                return False, "В одном из файлов нет результатов атрибутного анализа"

        X = project.radargram.distance_array
        Y = np.array([project.canal_number * 0.075] * len(X))
        Z = np.zeros(X.shape)
        coords = np.vstack((X, Y, Z)).T
        new_data_set = xr.Dataset({'amplitudes': (['trace', 'time'], data), 'coords': (['trace', 'coord'], coords)},
                                  coords={'time': project.radargram.time_array, 'coord': ['x', 'y', 'z'],
                                          'filename': file_name})
        return new_data_set

    def generate_slices(self, survey_file_name, time_delta, kind_data, special_coords_for_terrazond=False,
                        route_name=""):
        """генерирование амплитудных карт"""
        survey = Survey()
        survey.load(survey_file_name, is_open=False)
        if len(survey.project_file_names) == 0:
            return False, "В проекте нет ни одного файла"
        self.survey_file_name = survey.survey_file_name

        delta_distance = None
        delta_time = None
        time_max = 0
        all_data_set = None
        for i, file_name in enumerate(survey.project_file_names):
            self.events.procent_changing.emit(int(90 * i / len(survey.project_file_names)))
            new_data_set = self._generate_data_for_one_project(file_name, delta_time, delta_distance, time_max,
                                                               time_delta, kind_data)
            if i == 0:
                all_data_set = new_data_set
            else:
                all_data_set = xr.concat([all_data_set, new_data_set], dim='trace')

        X = all_data_set['coords'].loc[:, 'x'].values
        Y = all_data_set['coords'].loc[:, 'y'].values
        Z = all_data_set['coords'].loc[:, 'z'].values
        file_names = all_data_set.coords['filename'].values
        time_start = np.arange(0, time_max-time_delta, time_delta)
        amplitude = np.zeros([len(X), len(time_start)])
        colors = ["k"]*len(time_start)

        x_min = X.min(); x_max = X.max(); y_min = Y.min(); y_max = Y.max()
        self.slices_data = xr.Dataset({'file_names': (['trace'], file_names),
                                      'X': (['trace'], X),
                                      'Y': (['trace'], Y),
                                      'Z': (['trace'], Z),
                                      'amplitude': (['trace', 'timestart'], amplitude),
                                      'colors': (['timestart'], colors)},
                                      coords = {'timestart': time_start},
                                      attrs={'time_delta': time_delta,
                                             'time_max': time_max,
                                             'x_min': x_min,
                                             'x_max': x_max,
                                             'y_min': y_min,
                                             'y_max': y_max,
                                             'delta_distance': delta_distance,
                                             'amplitude_min': 0,
                                             'amplitude_max': 0,
                                             'slice_number': 0,
                                             'contrast': go.opts.amplitudemap.contrast,
                                             'is_consider_zero': go.opts.amplitudemap.is_consider_zero,
                                             'rough_detail': go.opts.amplitudemap.rough_detail,
                                             'fine_detail': go.opts.amplitudemap.fine_detail,
                                             'coef_detality': go.opts.amplitudemap.coef_detality,
                                             'is_amplitudemap_for_linear_project': go.opts.amplitudemap.is_amplitudemap_for_linear_project,
                                             'points_on_axis_count': go.opts.amplitudemap.points_on_axis_count,
                                             'mode': go.opts.amplitudemap.mode,
                                             'color_map': go.opts.amplitudemap.color_map,
                                             'min_hist_value': go.opts.amplitudemap.min_hist_value,
                                             'max_hist_value': go.opts.amplitudemap.max_hist_value,
                                             'is_absolute_mode_hist': go.opts.amplitudemap.is_absolute_mode_hist,
                                             'show_trajectories': go.opts.amplitudemap.show_trajectories,
                                             })

        for i, time1 in enumerate(time_start):
            self.events.procent_changing.emit(int(90 + 10 * i / len(time_start)))
            time2 = time1 + time_delta
            filter_data = all_data_set['amplitudes'].loc[:, time1:time2]
            mean_data = filter_data.mean(dim='time')
            self.slices_data['amplitude'].loc[:, time1] = mean_data
        self.slices_data.attrs['amplitude_min'] = self.slices_data['amplitude'].min().values
        self.slices_data.attrs['amplitude_max'] = self.slices_data['amplitude'].max().values

        self.save(survey_file_name)
        return True, ""

    def find_project(self, x, y, precision_x, precision_y):
        """поиск нужного файла по координатам трассы"""
        maskX = np.abs(self.slices_data["X"] - x) <= precision_x
        maskY = np.abs(self.slices_data["Y"] - y) <= precision_y
        X = self.slices_data["X"][maskX & maskY]
        Y = self.slices_data["Y"][maskX & maskY]
        file_names = self.slices_data["file_names"][maskX & maskY]

        if len(file_names) == 0:
            return None

        dist = np.sqrt((X-x)**2 + (Y-y)**2)
        ind = np.argmin(dist.values)

        file_name = str(file_names[ind].values)
        x_res = X[ind].values
        y_res = Y[ind].values

        X = self.slices_data["X"][self.slices_data["file_names"] == file_name]
        Y = self.slices_data["Y"][self.slices_data["file_names"] == file_name]
        maskX = np.abs(X - x_res) <= 0.0000000001
        maskY = np.abs(Y - y_res) <= 0.0000000001
        mask = maskX & maskY
        trace_number = np.argwhere(mask.values == True)[0][0]
        return file_name, trace_number, ind

    def get_slices_count(self):
        if self.slices_data is None:
            return None
        return len(self.slices_data.coords['timestart'])

    def get_slice_number(self):
        if self.slices_data is None:
            return None
        return self.slices_data.attrs['slice_number']

    def get_time_borders(self, slice_number):
        if slice_number >= len(self.slices_data.coords['timestart']):
            return None
        time1 = self.slices_data.coords['timestart'].values[slice_number]
        time2 = time1 + self.slices_data.attrs['time_delta']
        return time1, time2

    def find_slice_number(self, t):
        for slice_number in range(len(self.slices_data.coords['timestart'])):
            time1 = self.slices_data.coords['timestart'].values[slice_number]
            time2 = time1 + self.slices_data.attrs['time_delta']
            if time1 <= t <= time2:
                return slice_number
        return None

    def transform_amplitudes(self, amplitude, contrast=100, is_consider_zero=True):
        """преобразование амплитуд на амплитудных картах в зависимости от уровня контраста"""
        contrast = 100 - contrast

        amplitude_min = amplitude.min()
        amplitude_max = amplitude.max()

        if (amplitude_min > 0) or (amplitude_max < 0):
            is_consider_zero = False

        if is_consider_zero:
            amplitude_min_new = amplitude_min * contrast/100
            amplitude_max_new = amplitude_max * contrast/100
        else:
            amplitude_medium = (amplitude_min+amplitude_max)/2
            delta = (amplitude_medium-amplitude_min) * contrast/100
            amplitude_min_new = amplitude_medium - delta
            amplitude_max_new = amplitude_medium + delta

        amplitude_contrast = amplitude[:]
        amplitude_contrast[amplitude_contrast > amplitude_max_new] = amplitude_max_new
        amplitude_contrast[amplitude_contrast < amplitude_min_new] = amplitude_min_new
        amplitude_color = 255*((amplitude_contrast - amplitude_min_new) / (amplitude_max_new-amplitude_min_new))
        amplitude_color = amplitude_color.astype(int)
        return amplitude_color

    def add_point(self, X, Y):
        new_point = AmplitudeMapPoint(X,Y,"",self.get_slice_number())
        self.elements.append(new_point)
        self.selected_element = new_point
        return new_point

    def add_rectangle(self, X0, Y0, X1, Y1):
        new_rectangle = AmplitudeMapRectangle(X0, Y0, X1, Y1, "", self.get_slice_number())
        self.elements.append(new_rectangle)
        self.selected_element = new_rectangle
        return new_rectangle

    def add_text(self, X, Y, text_height):
        new_text = AmplitudeMapText(X, Y, text_height, "test", self.get_slice_number())
        self.elements.append(new_text)
        self.selected_element = new_text
        return new_text

    def add_polyline(self, vertexes, closed=False):
        new_polyline = AmplitudeMapPolyline(vertexes, closed, "", self.get_slice_number())
        self.elements.append(new_polyline)
        self.selected_element = new_polyline
        return new_polyline

    def delete_sel_element(self):
        self.elements = [el for el in self.elements if el != self.selected_element]
        self.selected_element = None

    def move_element(self, el, plus_x, plus_y):
        if (el.kind == "точка") or (el.kind == "текст"):
            el.X = el.X + plus_x
            el.Y = el.Y + plus_y
        elif el.kind == "прямоугольник":
            el.X0 = el.X0 + plus_x
            el.X1 = el.X1 + plus_x
            el.Y0 = el.Y0 + plus_y
            el.Y1 = el.Y1 + plus_y
        elif el.kind == "ломаная":
            for vertex in el.vertexes:
                vertex[0] = vertex[0] + plus_x
                vertex[1] = vertex[1] + plus_y

    def clear_elements(self):
        self.elements = []
        self.selected_element = None

    def clear_elements_on_slice(self):
        slice_num = self.get_slice_number()
        self.elements = [el for el in self.elements if el.slice_num != slice_num]
        self.selected_element = None

    def export_elements_to_csv(self, filename):
        rows = []
        for element in self.elements:
            if isinstance(element, AmplitudeMapPoint):
                row = [go.opts.translate("Точка"), element.X, element.Y]
            elif isinstance(element, AmplitudeMapRectangle):
                row = [go.opts.translate("Прямоугольник"), element.X0, element.Y0, element.X1, element.Y1]
            elif isinstance(element, AmplitudeMapText):
                row = [go.opts.translate("Текст"), element.X, element.Y]
            elif isinstance(element, AmplitudeMapPolyline):
                row = [go.opts.translate("Ломаная"), ] + list(element.vertexes.flatten())
            time1, time2 = self.get_time_borders(element.slice_num)
            row = [time1, time2] + row
            rows.append(row)

        with open(filename, "w", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow([go.opts.translate("t1, нс"), go.opts.translate("t2, нс"), go.opts.translate("вид элемента"), "X", "Y"])
            for row in rows:
                writer.writerow(row)

    def set_useful_area(self):
        points = []
        for element in self.elements:
            if isinstance(element, AmplitudeMapPoint) and element.visible:
                points.append([element.X, element.Y])
        if len(points) < 3:
            return
        self.useful_area_points = points

    def clear_useful_area(self):
        x_min, x_max, y_min, y_max = self.get_limits()
        self.useful_area_points = [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]

    def get_limits(self):
        x_min = self.slices_data.attrs["x_min"]
        x_max = self.slices_data.attrs["x_max"]
        y_min = self.slices_data.attrs["y_min"]
        y_max = self.slices_data.attrs["y_max"]
        return x_min, x_max, y_min, y_max


class AmplitudeMapEvents(QObject):
    """События, связанные с амплитудными картами"""
    procent_changing = pyqtSignal(int)  # изменение процента выполнения операции
