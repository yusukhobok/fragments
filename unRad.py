import logging
import math
import os
import struct
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, interpolate, stats
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.linalg import toeplitz
from scipy.signal import convolve2d, find_peaks
from skimage import img_as_float, img_as_ubyte
from skimage.exposure import equalize_hist
from skimage.feature import canny, greycoprops
from skimage.filters import sobel_v
from skimage.filters.rank import entropy, mean
from skimage.morphology import disk

import unGlobal
import unGPRCalculations
import unImageProcessing
from unGlobal import go
from unGPRCalculations import velocity_to_eps
from unTrace import Trace
from unUtils import Calculations, DateTransforms


class Rad:
    """Класс радарограммы"""

    def __init__(self, parent, data_trace, data_attribute, delta_distance, time_base, antenna_distance=1.0, default_velocity=0.100,
                 gpr_unit=None, antenna_name=None, frequency=1000.0, gain_coef_array=None, time_collecting=None,
                 trace_coordinate_array=None, surface_sample_array=None, is_z_shift_to_surface=False, pk_array=None,
                 underlays_file_names_data=None, underlays_file_names_attribute=None, start_position=None):
        self._parent = parent
        self._data_trace = data_trace  # матрица трасс [0..TraceCount-1, 0..samples_count-1]
        self.data_scheme = None  # уменьшенная копия радарограммы
        self._data_attribute = data_attribute  # результаты атрибутного анализа

        # файлы-подложки
        if underlays_file_names_data is None:
            self.underlays_file_names_data = []
        else:
            self.underlays_file_names_data = underlays_file_names_data
        if underlays_file_names_attribute is None:
            self.underlays_file_names_attribute = []
        else:
            self.underlays_file_names_attribute = underlays_file_names_attribute
        self.clear_remembered_underlays()

        # матрица экстремумов: 1й индекс - номер трассы; 2й индекс - вид экстремума: 0 - максимум, 1 - минимум, 2 - ноль; в ячейках - соответствующие значения времени
        self.peaks = []

        self.traces_count = self.data_trace.shape[0]  # количество трасс
        self.delta_distance = delta_distance  # шаг зондирования, м
        self.distance_max = (self.traces_count-1)*self.delta_distance  # длина профиля, м
        self.distance_array = np.linspace(0, self.distance_max, self.traces_count)  # расстояния от начала профиля на каждой остановке георадара, м

        self.samples_count = self.data_trace.shape[1]  # количество отсчетов на трассе
        self.time_base = time_base  # временная развертка, нс
        self.delta_time = float(self.time_base)/self.samples_count  # шаг дискретизации, нс
        self.time_max = self.time_base - self.delta_time  # максимальное время, нс
        self.time_array = np.linspace(0, self.time_max, self.samples_count)  # время каждого отсчета на трассе, нс

        self.gpr_unit = gpr_unit  # серия георадара
        self.antenna_name = antenna_name  # тип антенны
        self.frequency = frequency  # центральная частота сигнала, МГц

        self.time_collecting = time_collecting  # время каждой трассы

        if trace_coordinate_array is not None:
            self.trace_coordinate_array = trace_coordinate_array  # данные о координатах каждой трассы ['X' 'Y' 'Z']
        else:
            self.clear_coords()

        if surface_sample_array is not None:
            self.surface_sample_array = surface_sample_array  # положение поверхности для каждой трассы
        else:
            self.clear_surface_samples()
        self.is_z_shift_to_surface = is_z_shift_to_surface

        if pk_array is not None:
            if (np.all(pk_array[1:] - pk_array[:-1] > 0)) or (np.all(pk_array[1:] - pk_array[:-1] < 0)):
                self.pk_array = pk_array  # данные о пикете каждой трассы
            else:
                self.clear_pk_array(start_position)
        else:
            self.clear_pk_array(start_position)

        if gain_coef_array is not None:
            self.gain_coef_array = gain_coef_array # коэффициенты усиления по каждой трассе
        else:
            self.init_gain_coefs()

        self.antenna_distance = antenna_distance  # расстояние между антеннами, м (0 - если непостоянное)
        self.default_velocity = default_velocity  # скорость распространения волн по умолчанию, м/нс
        self.init_trace_velocity_array()
        self.generate_data_scheme()

        self.position = [0,0]  # текущая позиция на радарограмме (distance, time)

    parent = property()
    @parent.getter
    def parent(self):
        return self._parent
    @parent.setter
    def parent(self, value):
        if value is not None:
            self._parent = value
        else:
            self._parent = unGlobal.go.project
    @parent.deleter
    def parent(self):
        del self._parent

    data_trace = property()
    @data_trace.getter
    def data_trace(self):
        return self._data_trace
    @data_trace.setter
    def data_trace(self, value):
        if value is None:
            self._data_trace = None
        else:
            self._data_trace = np.nan_to_num(value)

    data_attribute = property()
    @data_attribute.getter
    def data_attribute(self):
        return self._data_attribute
    @data_attribute.setter
    def data_attribute(self, value):
        if value is None:
            self._data_attribute = None
        else:
            if value.shape != self.data_trace.shape:
                self._data_attribute = None
            else:
                self._data_attribute = np.nan_to_num(value)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_parent"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.parent = None

    def init_gain_coefs(self):
        self.gain_coef_array = np.zeros(self.time_array.shape)
        self.gain_coef_array.fill(1.0)

    def clear_coords(self):
        self.trace_coordinate_array = pd.DataFrame({'X': self.distance_array, 'Y': [0.0] * self.traces_count, 'Z': 0.0})

    def clear_surface_samples(self):
        self.surface_sample_array = np.zeros(self.traces_count, dtype=int)

    def clear_pk_array(self, start_position=None):
        if start_position is None:
            self.pk_array = self.distance_array
        else:
            self.pk_array = self.distance_array + start_position

    def init_trace_velocity_array(self):
        self.trace_velocity_array = np.empty(self.traces_count)
        self.trace_velocity_array.fill(self.default_velocity)

    def __refresh_time__(self):
        """Изменение оси времен"""
        self.samples_count = self.data_trace.shape[1]
        self.time_max = (self.samples_count - 1)*self.delta_time
        self.time_base = self.time_max + self.delta_time
        self.time_array = np.linspace(0, self.time_max, self.samples_count)
        self.init_gain_coefs()
        if self.position[1] > self.time_max:
            self.position[1] = self.time_max
        if self.position[1] not in self.time_array:
            self.position[1] = 0

    def __refresh_distance__(self, clear_data=True):
        """Изменение оси расстояний"""
        self.traces_count = self.data_trace.shape[0]
        self.distance_max = (self.traces_count-1)*self.delta_distance
        self.distance_array = np.linspace(0, self.distance_max, self.traces_count)
        if clear_data:
            self.parent.parts = None
            self.time_collecting = None
            self.clear_coords()
            self.clear_surface_samples()
            self.clear_pk_array()
            self.init_trace_velocity_array()
        if self.position[0] > self.distance_max:
            self.position[0] = self.distance_max
        if self.position[0] not in self.distance_array:
            self.position[0] = 0

    def refresh_data(self):
        self.data_trace = np.nan_to_num(self.data_trace)
        self.data_attribute = np.nan_to_num(self.data_attribute)

    def interpolate_data(self):
        source_data = self.data_trace * self.gain_coef_array
        data = interpolate.interp2d(self.time_array, self.distance_array, source_data, kind="linear")
        return data

    def interpolate_data_attribute(self):
        if self.data_attribute is None:
            return None
        else:
            source_data = self.data_attribute
            data = interpolate.interp2d(self.time_array, self.distance_array, source_data, kind="linear")
            return data

    def change_rad_properties(self, antenna_distance_new, default_velocity_new, gpr_unit_new, antenna_name_new,
                              frequency_new):
        """Изменение параметров радарограммы"""
        self.antenna_distance = antenna_distance_new
        self.default_velocity = default_velocity_new
        self.gpr_unit = gpr_unit_new
        self.antenna_name = antenna_name_new
        self.frequency = frequency_new

    def change_velocity(self, default_velocity_new):
        self.default_velocity = default_velocity_new

    def change_delta_distance_and_delta_time(self, delta_distance_new, delta_time_new):
        if self.delta_distance!=delta_distance_new or self.delta_time!=delta_time_new:
            f_data = RectBivariateSpline(self.distance_array, self.time_array, self.data_trace)
            traces_count_new = int(round(self.distance_max/delta_distance_new)+1)
            samples_count_new = int(round(self.time_max/delta_time_new)+1)
            data_new = np.empty([traces_count_new, samples_count_new])
            for i in range(traces_count_new):
                go.events.procent_changing.emit(int(100 * i / traces_count_new))
                for j in range(samples_count_new):
                    distance = i*delta_distance_new
                    t = j*delta_time_new
                    data_new[i,j] = f_data(distance,t)
            self.data_trace = data_new

            if self.delta_distance != delta_distance_new:
                self.delta_distance = delta_distance_new
                self.__refresh_distance__()

            if self.delta_time != delta_time_new:
                self.delta_time = delta_time_new
                self.__refresh_time__()

            self.generate_data_scheme()
            self.position = [0, 0]
            self.parent.pop_binding = None

            for el in self.parent.elements.data:
                if el.kind == "ломаная":
                    el.calc_vertexes_all()

    def clear_remembered_underlays(self):
        self.underlays_file_names_data = [file_name for file_name in self.underlays_file_names_data if os.path.exists(file_name)]
        self.underlays_file_names_attribute = [file_name for file_name in self.underlays_file_names_attribute if os.path.exists(file_name)]

    def delete_remembered_underlay_data(self, deleted_file_name):
        self.underlays_file_names_data = [file_name for file_name in self.underlays_file_names_data if file_name != deleted_file_name]

    def delete_remembered_underlay_attribute(self, deleted_file_name):
        self.underlays_file_names_attribute = [file_name for file_name in self.underlays_file_names_attribute if file_name != deleted_file_name]

    def save_underlay(self, file_name, is_attribute=False, is_remember=True):
        with open(file_name, "wb") as f:
            f.write(struct.pack("ii", self.traces_count, self.samples_count))
            if not is_attribute:
                self.data_trace.tofile(f)
            else:
                if self.data_attribute is not None:
                    self.data_attribute.tofile(f)
            if is_remember:
                if not is_attribute:
                    if file_name not in self.underlays_file_names_data:
                        self.underlays_file_names_data.append(file_name)
                else:
                    if file_name not in self.underlays_file_names_attribute:
                        self.underlays_file_names_attribute.append(file_name)

    def load_underlay(self, file_name, is_attribute=False):
        if not os.path.exists(file_name):
            return False
        with open(file_name, "rb") as f:
            part_of_file = f.read(struct.calcsize("ii"))
            traces_count, samples_count = struct.unpack("ii", part_of_file)
            if traces_count != self.traces_count or samples_count != self.samples_count:
                return False
            data = np.fromfile(f, count=traces_count * samples_count)
            try:
                data = np.reshape(data, (traces_count, samples_count))
            except ValueError:
                return False
            data = np.nan_to_num(data)
            if not is_attribute:
                self.data_trace = data
                self.data_attribute = None
            else:
                self.data_attribute = data
        return True

    def import_from_npy(self, file_name):
        data = np.load(file_name)
        traces_count, samples_count = data.shape
        if traces_count != self.traces_count or samples_count != self.samples_count:
            return False
        self.data_attribute = data
        return True

    def cut_data(self, count_gaps_traces, count_gaps_samples):
        if (count_gaps_traces < 1) or (count_gaps_samples < 1):
            return
        if (count_gaps_traces == 1) and (count_gaps_samples == 1):
            return
        if (count_gaps_traces > self.traces_count/2) or (count_gaps_samples > self.samples_count/2):
            return

        self.data_trace = self.data_trace[::count_gaps_traces, ::count_gaps_samples]
        if self.data_attribute is not None:
            self.data_attribute = self.data_attribute[::count_gaps_traces, ::count_gaps_samples]
        self.peaks = []

        self.traces_count = self.data_trace.shape[0]
        self.delta_distance = self.distance_max/(self.traces_count-1)
        self.distance_array = np.linspace(0, self.distance_max, self.traces_count)
        self.samples_count = self.data_trace.shape[1]
        self.delta_time = float(self.time_max) / (self.samples_count-1)
        self.time_array = np.linspace(0, self.time_max, self.samples_count)
        self.time_base = self.time_max + self.delta_time

        if self.time_collecting is not None:
            self.time_collecting = self.time_collecting[::count_gaps_traces]

        self.trace_coordinate_array = self.trace_coordinate_array.iloc[::count_gaps_traces]
        self.pk_array = self.pk_array[::count_gaps_traces]
        self.clear_surface_samples()

        self.gain_coef_array = self.gain_coef_array[::count_gaps_samples]
        self.trace_velocity_array = self.trace_velocity_array[::count_gaps_traces]
        self.position = [0, 0]
        self.parent.pop_binding = None
        for element in self.parent.elements.data:
            if element.kind == "ломаная":
                element.calc_vertexes_all()

    def change_delta_distance(self, delta_distance_new):
        distance_max_old = self.distance_max
        if delta_distance_new <= 0:
            return
        self.delta_distance = delta_distance_new
        self.__refresh_distance__(clear_data=False)
        self.position = [0, 0]
        self.parent.elements.change_elements_after_change_distance_max(distance_max_old, self.distance_max)

    def change_distance_max(self, distance_max_new: float):
        if distance_max_new <= 0:
            return
        distance_max_old = self.distance_max
        self.distance_max = distance_max_new
        self.delta_distance = self.distance_max / (self.traces_count-1)
        self.__refresh_distance__(clear_data=False)
        self.position = [0, 0]
        self.parent.elements.change_elements_after_change_distance_max(distance_max_old, distance_max_new)

    def change_time_max(self, time_max_new: float):
        if time_max_new <= 0:
            return
        self.time_max = time_max_new
        self.delta_time = self.time_max / (self.samples_count - 1)
        self.__refresh_time__()

    def set_pickets(self, start, finish, is_end):
        if not is_end:
            finish = start + self.distance_max
        if np.abs(start - finish) < 1e-5:
            return
        newLmax = np.abs(finish - start)
        if np.abs(self.distance_max - newLmax) > 1e-5:
            self.change_distance_max(newLmax)
        if start < finish:
            self.pk_array = start + self.distance_array
        else:
            self.pk_array = start - self.distance_array

    def set_pickets_by_start_distance(self, distance0):
        self.pk_array = self.distance_array + distance0

    def correct_amplitudes(self):
        """Корректировка амплитуд"""
        self.to_zero()
        DataAbs = np.abs(self.data_trace)
        mean = np.mean(DataAbs)
        limit = 100 * mean
        self.data_trace[DataAbs > limit] = 0

    def to_zero(self):
        """Приведение трасс радарограммы к нулю"""
        av = self.calc_average()
        av = av[:, np.newaxis]
        self.data_trace = self.data_trace - av

    def rectify(self):
        """Взятие амплитуд по модулю"""
        self.data_trace = np.abs(self.data_trace)

    def smoothing(self, window_len):
        """Сглаживание"""
        if window_len % 2 == 0:
            window_len = window_len + 1
        for i in range(self.data_trace.shape[0]):
            x = np.array(self.data_trace[i])
            y = Calculations.savitzky_golay(x, window_len, 3)
            self.data_trace[i] = y

    def smoothing_minus(self, window_len):
        """Коррекция дрейфа нуля"""
        if window_len % 2 == 0:
            window_len = window_len + 1
        for i in range(self.data_trace.shape[0]):
            go.events.procent_changing.emit(int(100 * i / self.traces_count))
            x = np.array(self.data_trace[i])
            y = Calculations.savitzky_golay(x, window_len, 3)
            z = x-y
            self.data_trace[i] = z

    def average_horizontal(self, win_len):
        """Усреднение по горизонтали с вычитанием"""
        if win_len == 100:
            mean = np.mean(self.data_trace, axis=0)
            self.data_trace -= mean[np.newaxis,:]
        else:
            win_traces = int(self.traces_count * win_len / 100)
            NewData = np.zeros((self.traces_count, self.samples_count))
            for i in range(self.traces_count):
                go.events.procent_changing.emit(int(100 * i / self.traces_count))
                i1 = max(0, i - win_traces//2)
                i2 = min(self.traces_count-1, i + win_traces // 2)
                mean = np.mean(self.data_trace[i1:i2+1, :], axis=0)
                NewData[i,:] = self.data_trace[i,:] - mean
            self.data_trace = NewData

    def normalization_vertical(self, window_len, kind=0):
        """Нормировка амплитуд по вертикали"""
        if kind == 0:
            i = self.find_nearest_trace(self.position[0])
            amplitude = self.data_trace[i,:]
            trace = Trace(self.time_array, amplitude, "")
            (_, result, _, _, _, _, _) = trace.abs_trace(window_len)
            self.gain_coef_array = result/amplitude
            self.gain_coef_array[np.isnan(self.gain_coef_array)] = 0
        else:
            amplitude = self.mean_trace()
            trace = Trace(self.time_array, amplitude, "")
            (_, result, _, _, _, _, _) = trace.abs_trace(window_len)
            self.gain_coef_array = result/amplitude
            self.gain_coef_array[np.isnan(self.gain_coef_array)] = 0

    def normalization_horizontal(self, coef_normalize):
        """Нормализация амплитуд по горизонтали"""

        #Суммы амплитуд для каждой трассы
        Asum = np.sum(abs(self.data_trace), 1)

        #p - положение оси годографа
        if self.parent.ground_model.is_model:
            p = self.find_nearest_trace(self.parent.ground_model.axe_position)
        else:
            p = 0

        if coef_normalize == 0:
            #Если коэффициент нормализации равен нулю, то все для каждой трассы сумма амплитуд приводится к сумме амплитуд по оси годографа
            for i in range(self.traces_count):
                if Asum[i] != 0:
                    k = Asum[p] / Asum[i]
                    self.data_trace[i,:] = self.data_trace[i,:] * k
        else:
            if self.parent.ground_model.is_model:
                distance0 = self.parent.ground_model.axe_position
                distance_max = max(distance0, self.distance_max-distance0)
                K = coef_normalize
                for i in range(self.traces_count):
                    kp = (K-1)*abs(self.distance_array[i] - distance0)/distance_max + 1
                    k = (Asum[p] / Asum[i]) / kp
                    self.data_trace[i,:] = self.data_trace[i,:] * k
                np.nan_to_num(self.data_trace, copy=False)
        self.init_gain_coefs()

    def gain(self, gainFunction, coefExp, leftCoefLine, rightCoefLine):
        if gainFunction == 0: #прямая
            x = [self.time_array[0], self.time_array[-1]]
            y = [leftCoefLine, rightCoefLine]
            self.gain_coef_array = np.interp(self.time_array, x, y)
        else: #экспонента
            q = self.time_array[-1] * (101 - coefExp) / 200
            self.gain_coef_array = np.exp(-(self.time_array[-1] - self.time_array) / q)

    def filtrate(self, mask):
        """Частотная фильтрация"""
        for i in range(self.traces_count):
            trace = Trace(self.time_array, self.data_trace[i, :], "")
            (_, result, _, _, _, _, _) = trace.filtrate(mask)
            self.data_trace[i, :] = result

    def calc_spectrum(self, trace):
        spectrum = np.fft.fft(trace.trace)
        length = len(trace.trace)
        dt = trace.time_array[1] - trace.time_array[0]

        f = np.arange(length)/(dt*length)
        ps = np.abs(spectrum) / length

        half_count = int(np.floor(len(f) / 2))
        half_f = np.array(f[:half_count])
        half_ps = np.array(ps[:half_count])
        return half_f, half_ps

    def calc_average(self):
        return np.mean(self.data_trace[:, -int(self.data_trace.shape[1] * go.opts.processing.samples_count_for_zero):], 1)

    def mean_trace(self):
        """Поиск средней трассы"""
        return np.average(self.data_trace, 0)

    def reflect(self):
        """Отражение радарограммы"""
        self.data_trace = np.flipud(self.data_trace)
        self.trace_coordinate_array = self.trace_coordinate_array.iloc[::-1]
        self.trace_coordinate_array = self.trace_coordinate_array.reset_index(drop=True)
        self.surface_sample_array = np.flip(self.surface_sample_array)
        self.time_collecting = np.flip(self.time_collecting)

    def straighten_polyline(self, polyline):
        """Искажение радарограммы с целью распрямления границы"""
        if polyline.kind != "ломаная":
            return False
        if np.isnan(polyline.vertexes_all[0]):
            return False

        t0 = polyline.vertexes_all[0]
        k0 = self.find_nearest_sample(t0)

        A_mean = np.mean(self.data_trace)
        for i in range(self.traces_count):
            t = polyline.vertexes_all[i]
            if not np.isnan(t):
                k = self.find_nearest_sample(t)
                if k < k0:
                    dk = k0 - k
                    self.data_trace[i][dk:] = self.data_trace[i][:-dk]
                    self.data_trace[i][:dk] = A_mean
                elif k > k0:
                    dk = k - k0
                    self.data_trace[i][:-dk] = self.data_trace[i][dk:]
                    self.data_trace[i][-dk:] = A_mean
        self.data_attribute = None
        self.clear_surface_samples()
        return True

    def clear_upper_polyline(self, polyline):
        """Очистка данных выше границы"""
        if polyline.kind != "ломаная":
            return False
        if np.isnan(polyline.vertexes_all[0]):
            unGlobal.message_error(go.opts.translate("Граница должна начинаться с самого начала радарограммы. Очистка невозможна"),
                                   about_developers=False)
            return False
        for i in range(self.traces_count):
            t = polyline.vertexes_all[i]
            if np.isnan(t):
                continue
            j = self.find_nearest_sample(t)
            self.data_trace[i,:j] = 0
        return True

    def trim_above_polyline(self, polyline):
        """Обрезка радарограммы выше выделенной границы"""
        if polyline.kind != "ломаная":
            return

        A_mean = np.mean(self.data_trace)
        for i in range(self.traces_count):
            t = polyline.vertexes_all[i]
            if not np.isnan(t):
                k = self.find_nearest_sample(t)
                self.data_trace[i][:-k-1] = self.data_trace[i][k+1:]
                self.data_trace[i][-k:] = A_mean
        self.data_attribute = None
        self.clear_surface_samples()

    def trim_from_start(self, time):
        """Обрезка начала радарограммы (до времени time)"""
        num_sample = self.find_nearest_sample(time)
        if (num_sample is not None) and (num_sample < self.samples_count-5):
            self.data_trace = self.data_trace[:,num_sample:]
            self.gain_coef_array = self.gain_coef_array[num_sample:]
            self.__refresh_time__()
            self.data_attribute = None
            self.clear_surface_samples()
            return True
        else:
            return False

    def trim_to_end(self, time):
        """Обрезка конца радарограммы (от времени time)"""
        num_sample = self.find_nearest_sample(time)
        if (num_sample is not None) and (num_sample > 5):
            self.data_trace = self.data_trace[:,:num_sample]
            self.gain_coef_array = self.gain_coef_array[:num_sample]
            self.__refresh_time__()
            self.data_attribute = None
            return True
        else:
            return False

    def trim_from_left(self, distance):
        """Обрезка начала радарограммы слева"""
        num_trace = self.find_nearest_trace(distance)
        if (num_trace is not None) and (num_trace < self.traces_count-5):
            self.data_trace = self.data_trace[num_trace:,:]
            self.__refresh_distance__(clear_data=False)
            self.surface_sample_array = self.surface_sample_array[num_trace:]
            self.pk_array = self.pk_array[num_trace:]
            self.trace_velocity_array = self.trace_velocity_array[num_trace:]
            self.trace_coordinate_array = self.trace_coordinate_array.iloc[num_trace:]
            if self.time_collecting is not None:
                self.time_collecting = self.time_collecting[num_trace:]

            self.data_attribute = None
            return True
        else:
            return False

    def trim_to_right(self, distance):
        """Обрезка конца радарограммы справа"""
        num_trace = self.find_nearest_trace(distance)
        if (num_trace is not None) and (num_trace > 5):
            self.data_trace = self.data_trace[:num_trace,:]
            self.__refresh_distance__(clear_data=False)
            self.surface_sample_array = self.surface_sample_array[:num_trace]
            self.pk_array = self.pk_array[:num_trace]
            self.trace_velocity_array = self.trace_velocity_array[:num_trace]
            self.trace_coordinate_array = self.trace_coordinate_array.iloc[:num_trace]
            if self.time_collecting is not None:
                self.time_collecting = self.time_collecting[:num_trace]
            self.data_attribute = None
            return True
        else:
            return False

    def del_trace(self, distance):
        """Удаление трассы на расстоянии distance"""
        num_trace = self.find_nearest_trace(distance)
        if num_trace is not None:
            if not go.opts.processing.delete_trace_with_interpolate:
                self.data_trace = np.delete(self.data_trace, num_trace, 0)
                self.__refresh_distance__()
                self.parent.elements.clear_elements()
            else:
                if num_trace == 0:
                    self.data_trace[num_trace,:] = self.data_trace[num_trace+1,:]
                elif num_trace == self.traces_count-1:
                    self.data_trace[num_trace,:] = self.data_trace[num_trace-1,:]
                else:
                    self.data_trace[num_trace,:] = (self.data_trace[num_trace-1,:] + self.data_trace[num_trace+1,:])/2

    def del_part_traces(self, distance1, distance2, time1, time2):
        """Удаление части радарограммы"""
        L1_ = min(distance1, distance2)
        L2_ = max(distance1, distance2)
        time1_ = min(time1, time2)
        time2_ = max(time1, time2)

        n1 = self.find_nearest_trace(L1_)
        n2 = self.find_nearest_trace(L2_)
        m1 = self.find_nearest_sample(time1_)
        m2 = self.find_nearest_sample(time2_)
        mean_trace = np.mean(self.calc_average())
        if n1 is not None and n2 is not None:
            self.data_trace[n1:n2,m1:m2] = mean_trace
        self.data_attribute = None

    def insert_trace(self, distance1, distance2):
        """Вставка пустых трасс"""
        L1_ = min(distance1, distance2)
        L2_ = max(distance1, distance2)
        if L2_ >= L1_:
            i = self.find_nearest_trace(L1_)
            N = int((L2_-L1_)/self.delta_distance)+1
            new_data = np.zeros((N, self.samples_count))
            self.data_trace = np.vstack((self.data_trace[:i,:], new_data, self.data_trace[i:,:]))
            self.__refresh_distance__()
            self.parent.elements.clear_elements()
        self.data_attribute = None
        self.parent.pop_binding = None

    def compensate_time_delta(self, time):
        N = self.find_nearest_sample(time)
        new_data = np.zeros((self.traces_count, N))
        self.data_trace = np.hstack((new_data, self.data_trace))
        self.__refresh_time__()
        self.data_attribute = None
        self.parent.pop_binding = None

    def dewow(self):
        for i in range(self.traces_count):
            go.events.procent_changing.emit(int(100 * i / self.traces_count))
            trace = Trace(self.time_array, self.data_trace[i,:], "")
            (_, result, _, _, _, _, _) = trace.dewow()
            self.data_trace[i, :] = result

    def equalize_hist(self):
        data = self.data_trace * self.gain_coef_array
        data = equalize_hist(data)
        self.data_trace = data
        self.init_gain_coefs()
        self.to_zero()

    def mean_as_image(self):
        data = self.data_trace * self.gain_coef_array
        data = (data - data.min()) / (data.max() - data.min())
        data = img_as_float(data)
        data = mean(data, disk(5))
        self.data_trace = data
        self.init_gain_coefs()
        self.to_zero()

    def cut_direct_signal(self, time0, time1, distance):
        num0 = self.find_nearest_sample(time0)
        num1 = self.find_nearest_sample(time1)
        num_trace = self.find_nearest_trace(distance)
        trace = np.copy(self.data_trace[num_trace, num0:num1 + 1])
        self.data_trace[:, num0:num1 + 1] -= trace

    def hod_transform(self, axes_position):
        """Трансформация годографа (учет прохождения прямой воздушной волны)"""
        if self.traces_count > 1000:
            return

        if axes_position < 0 or axes_position > self.distance_max:
            return
        c = 0.3

        tmax = max(axes_position, self.distance_max - axes_position)/c
        sample_max = round(tmax/self.delta_time)
        new_data_trace = np.zeros((self.traces_count, self.samples_count + sample_max))

        for i in range(0,self.traces_count):
            distance = i*self.delta_distance
            t = abs(axes_position - distance)/c

            sample = round(t/self.delta_time)
            sample2 = sample_max - sample

            if sample2 == 0:
                new_data_trace[i, sample:] = self.data_trace[i, :]
            else:
                new_data_trace[i, sample:-sample2] = self.data_trace[i, :]

        self.data_trace = new_data_trace
        self.__refresh_time__()

    def hod_untransform(self, axes_position):
        """Отмена трансформации годографа"""
        if self.traces_count > 1000:
            return

        if axes_position < 0 or axes_position > self.distance_max:
            return
        c = 0.3

        tmax = max(axes_position, self.distance_max - axes_position)/c
        sample_max = round(tmax/self.delta_time)
        new_data_trace = np.zeros((self.traces_count, self.samples_count - sample_max))

        for i in range(0,self.traces_count):
            distance = i*self.delta_distance
            t = abs(axes_position - distance)/c

            sample = round(t/self.delta_time)
            sample2 = sample_max - sample

            if sample2 == 0:
                new_data_trace[i, :] = self.data_trace[i, sample:]
            else:
                new_data_trace[i, :] = self.data_trace[i, sample:-sample2]

        self.data_trace = new_data_trace
        self.__refresh_time__()

    def prepare_attributes_before_analysis(self):
        if not go.opts.processing.is_fast_attribute_analysis:
            self.data_attribute = np.zeros(self.data_trace.shape)
        else:
            if self.data_attribute is None:
                self.data_attribute = self.data_trace * self.gain_coef_array

        if go.opts.processing.is_fast_attribute_analysis:
            (start_trace, finish_trace, start_sample, finish_sample) = self.find_window_for_attribute_analysis()
            i_range = list(range(start_trace, finish_trace))
            j_range = list(range(start_sample, finish_sample))
        else:
            i_range = list(range(self.traces_count))
            j_range = list(range(self.samples_count))
        return i_range, j_range

    def split_into_levels(self, levels_count):
        """Разбиение амплитуд по уровням"""
        if levels_count == 1:
            mask = self.data_trace >= 0
            mask = mask.astype(int)
            self.data_attribute = mask
            self.data_attribute = self.data_attribute.astype(float)
        else:
            i_range, j_range = self.prepare_attributes_before_analysis()
            data = self.data_trace * self.gain_coef_array

            k = levels_count
            av = np.average(data)
            amplitude_min = np.min(data)
            amplitude_max = np.max(data)
            for i in i_range:
                print("%d из %d" % (i, self.traces_count))
                go.events.procent_changing.emit(100 * (i - i_range[0]) / len(i_range))
                l = max(av-amplitude_min, amplitude_max-av)
                dl = l/k
                for j in j_range:
                    if data[i,j] > av:
                        x = av
                        while data[i,j] > x:
                            x = x + dl
                        self.data_attribute[i,j] = x
                    else:
                        x = av
                        while x > data[i,j]:
                            x = x - dl
                        self.data_attribute[i,j] = x

    def find_breaks(self, autofind_min_amplitude):
        """Поиск переломов сигнала"""
        i_range, j_range = self.prepare_attributes_before_analysis()
        data = self.data_trace * self.gain_coef_array

        for i in i_range:
            go.events.procent_changing.emit(int(100 * (i - i_range[0]) / len(i_range)))
            trace = Trace(self.time_array, data[i,:], "")
            DerTime, DerTrace, S1, S2, S3, S4, S5 = trace.derivative()
            DerTrace = Trace(DerTime, DerTrace, "")
            time, amplitude = DerTrace.conversion_zero()
            for (t,a) in zip(time, amplitude):
                if a > autofind_min_amplitude:
                    j = self.find_nearest_sample(t)
                    self.data_attribute[i][j] = a

    def find_derivative(self):
        """Вычисление производной сигнала"""
        i_range, j_range = self.prepare_attributes_before_analysis()
        data = self.data_trace * self.gain_coef_array
        for i in i_range:
            self.data_attribute[i,j_range] = np.append(np.diff(data[i,j_range]), 0)

    def find_extremums_range(self):
        """Расчет размаха экстремумов"""
        i_range, j_range = self.prepare_attributes_before_analysis()
        data = self.data_trace * self.gain_coef_array

        for i in i_range:
            print("%d из %d" % (i, self.traces_count))
            go.events.procent_changing.emit(int(100 * (i - i_range[0]) / len(i_range)))
            trace = Trace(self.time_array[j_range], data[i,j_range], "")
            (_, result, _, _, _, _, _) = trace.find_extremums_range(False)
            self.data_attribute[i, j_range] = result

    def attenuation_field(self):
        """Расчет поля затухания"""
        i_range, j_range = self.prepare_attributes_before_analysis()
        data = self.data_trace * self.gain_coef_array

        for i in i_range:
            print("%d из %d" % (i, self.traces_count))
            go.events.procent_changing.emit(int(100 * (i - i_range[0]) / len(i_range)))
            trace = Trace(self.time_array[j_range], data[i,j_range], "")
            (_, result, _, _, _, _, _) = trace.find_attenuation()
            self.data_attribute[i, j_range] = result

    def find_window_for_attribute_analysis(self):
        width = self.find_nearest_trace(go.opts.processing.fast_attribute_analysis_dl)
        height = self.find_nearest_sample(go.opts.processing.fast_attribute_analysis_dt)
        i = self.find_nearest_trace(self.position[0])
        j = self.find_nearest_sample(self.position[1])

        start_trace = max(0, i-width)
        finish_trace = min(self.traces_count, i+width)
        start_sample = max(0, j-height)
        finish_sample = min(self.samples_count, j+height)
        return start_trace, finish_trace, start_sample, finish_sample

    def hilbert(self, code):
        """Преобразование Гильберта; code: 0 - огибающая, 1 - мгновенная фаза, 2 - мгновенная частота"""
        i_range, j_range = self.prepare_attributes_before_analysis()
        data = self.data_trace * self.gain_coef_array
        for i in i_range:
            go.events.procent_changing.emit(int(100 * (i - i_range[0]) / len(i_range)))
            trace = Trace(self.time_array[j_range], data[i,j_range], "")
            (_, result, _, _, _, _, _) = trace.hilbert(code)
            if code == 2:
                result = np.append(result, 0)
            self.data_attribute[i, j_range] = result

    def evaluate_time_for_cosine_phase(self, M):
        data = self.data_trace * self.gain_coef_array
        if go.opts.processing.is_fast_attribute_analysis:
            (start_trace, finish_trace, start_sample, finish_sample) = self.find_window_for_attribute_analysis()
            count = finish_trace - start_trace
        else:
            count = self.traces_count
        time1 = time.time()
        trace = Trace(self.time_array, data[0, :], "")
        (_, result, _, _, _, _, _) = trace.cosine_phase(M)
        time2 = time.time()
        delta = ((time2-time1)*count)/60
        return delta

    def cosine_phase(self, M):
        """Преобразование cosine phase [Dossi 2015]"""
        i_range, j_range = self.prepare_attributes_before_analysis()
        data = self.data_trace * self.gain_coef_array

        for i in i_range:
            print("%d из %d" % (i, self.traces_count))
            go.events.procent_changing.emit(int(100 * (i - i_range[0]) / len(i_range)))
            trace = Trace(self.time_array[j_range], data[i,j_range], "")
            (_, result, _, _, _, _, _) = trace.cosine_phase(M)
            self.data_attribute[i, j_range] = result

    def filter_sobel(self):
        """Применение оператора Собеля"""
        i_range, j_range = self.prepare_attributes_before_analysis()
        data = self.data_trace * self.gain_coef_array
        data = (data - data.min()) / (data.max() - data.min())
        data = img_as_float(data)
        data = equalize_hist(data)

        if go.opts.processing.is_fast_attribute_analysis:
            (start_trace, finish_trace, start_sample, finish_sample) = self.find_window_for_attribute_analysis()
            data = data[start_trace:finish_trace, start_sample:finish_sample]

        data = sobel_v(data)
        if go.opts.processing.is_fast_attribute_analysis:
            self.data_attribute[start_trace:finish_trace, start_sample:finish_sample] = data
        else:
            self.data_attribute = data

    def binarization_otsu(self, qimage):
        """Бинаризация (порог вычисляется по методу Оцу)"""
        logging.info("binarization_otsu 1")
        binary = unImageProcessing.binarization_otsu(qimage)
        logging.info("binarization_otsu 10")
        self.data_attribute = binary.astype(float)
        logging.info("binarization_otsu 11")

    def entropy(self):
        """Расчет энтропии"""
        i_range, j_range = self.prepare_attributes_before_analysis()
        data = self.data_trace * self.gain_coef_array
        data = (data - data.min()) / (data.max() - data.min())
        data = img_as_float(data)
        data = equalize_hist(data)

        if go.opts.processing.is_fast_attribute_analysis:
            (start_trace, finish_trace, start_sample, finish_sample) = self.find_window_for_attribute_analysis()
            data = data[start_trace:finish_trace, start_sample:finish_sample]

        data = entropy(data, disk(10))
        if go.opts.processing.is_fast_attribute_analysis:
            self.data_attribute[start_trace:finish_trace, start_sample:finish_sample] = data
        else:
            self.data_attribute = data

    def calc_resolution(self, velocity, frequency, height):
        frequency = frequency / 1000
        WaveLength = velocity / frequency
        dY = 0.5*WaveLength
        eps = velocity_to_eps(self.default_velocity)
        dX = WaveLength/4 + height/np.sqrt(eps-1)
        return {"HorResolution": dX, "VertResolution": dY}

    def set_position(self, distance=None, time=None):
        if distance is not None:
            if go.opts.processing.round_l_to_trace:
                self.position[0] = self.round_distance(distance)
            else:
                self.position[0] = distance
        if time is not None:
            if go.opts.processing.round_t_to_sample:
                self.position[1] = self.round_time(time)
            else:
                self.position[1] = time

    def find_coords_of_first_and_last_traces(self):
        return self.trace_coordinate_array.values[[0, -1]]

    def find_trace_by_coord(self, x, y, z=None):
        """Поиск номера трассы, наиболее близкой к заданным координатам"""
        min_dist = np.inf
        min_trace = None
        X = self.trace_coordinate_array["X"].values
        Y = self.trace_coordinate_array["Y"].values
        Z = self.trace_coordinate_array["Z"].values
        for i in range(self.traces_count):
            if z is not None:
                dist = np.sqrt((x-X[i]) ** 2 + (y-Y[i]) ** 2 + (z-Z[i]) ** 2)
            else:
                dist = np.sqrt((x-X[i]) ** 2 + (y-Y[i]) **2 )
            if min_dist > dist:
                min_dist = dist
                min_trace = i
        return min_trace

    def find_trace_by_one_coord(self, value, kind="X"):
        """Поиск номера трассы, наиболее близкой к заданной координате (или пикету, или километру)"""
        if (kind == "PK") or (kind == "KM"):
            coords = self.pk_array
        else:
            coords = self.trace_coordinate_array[kind].values
        min_dist = np.inf
        min_trace = None
        for i in range(go.project.radargram.traces_count):
            dist = np.abs(value-coords[i])
            if min_dist > dist:
                min_dist = dist
                min_trace = i
        return min_trace

    def find_nearest_traces(self, distance):
        """Поиск номеров трасс, ближайшей к расстояниям distance (distance - массив numpy)"""
        if type(distance) is not np.ndarray:
            return None
        distance[distance < 0] = 0
        distance[distance > self.distance_max] = self.distance_max
        ind = np.round(distance/self.delta_distance).astype(int)
        return ind

    def find_nearest_trace(self, distance):
        """Поиск трассы, ближайшей к расстоянию distance"""
        if distance < 0:
            return 0
        elif distance > self.distance_max:
            return self.traces_count-1
        else:
            return int(np.round(distance/self.delta_distance))

    def find_nearest_samples(self, time_array):
        """Поиск номеров временных отсчетов, ближайшей к временам time (time - массив numpy)"""
        if type(time_array) is not np.ndarray:
            return None
        time_array[time_array < 0] = 0
        time_array[time_array > self.time_max] = self.time_max
        ind = np.round(time_array/self.delta_time).astype(int)
        return ind

    def find_nearest_sample(self, t):
        """Поиск отсчета, ближайшего к времени t"""
        assert not np.isnan(t), "t = NAN!"
        if t < 0:
            return 0
        elif t > self.time_max:
            return self.samples_count-1
        else:
            # return np.argmin([abs(Ti - t) for Ti in self.time_array])
            return int(np.round(t/self.delta_time))

    def round_distance(self, distance):
        """Округление расстояния до ближайшей трассы"""
        k = self.find_nearest_trace(distance)
        return self.distance_array[k]

    def round_time(self, t):
        """Округление времени до ближайшего отсчета"""
        k = self.find_nearest_sample(t)
        return self.time_array[k]
