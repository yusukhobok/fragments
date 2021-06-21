import datetime
import decimal
import hashlib
import math
import os.path
from math import factorial

import numpy as np
import pandas as pd
import utm
from numba import jit
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree as KDTree
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from unGlobal import go


class Geometry:
    @staticmethod
    def offset_segment(x1, y1, x2, y2, offset):
        """Сдвижка отрезка"""
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        x1p = x1 + offset * (y2 - y1) / distance
        x2p = x2 + offset * (y2 - y1) / distance
        y1p = y1 + offset * (x1 - x2) / distance
        y2p = y2 + offset * (x1 - x2) / distance
        return x1p, y1p, x2p, y2p

    @staticmethod
    def is_same_point(p1, p2):
        """Проверка двух точек на совпадение (с заданной точностью)"""
        return np.hypot(p1[0] - p2[0], p1[1] - p2[1]) <= go.opts.interface.pick_precision

    @staticmethod
    def distance_between_points(p1, p2):
        """Расстояние между двумя точками"""
        return np.hypot(p1[0]-p2[0], p1[1]-p2[1])

    @staticmethod
    def is_point_in_polygon(point, polygon_points):
        """Проверка, находится ли точка внутри многоугольника"""
        point = Point(point)
        polygon = Polygon(polygon_points)
        return polygon.contains(point)


class Picketazh:
    @staticmethod
    def transform_dist_to_picket(distance):
        "Преобразование расстояния в пикетаж"
        pk_array = int(np.trunc(distance / 100))
        Plus = distance - 100 * pk_array
        return pk_array, Plus

    @staticmethod
    def transform_picket_to_dist(pk_array, Plus):
        "Преобразование пикетажа в расстояние"
        if pk_array < 0:
            return
        if Plus < 0:
            return
        if Plus >= 100:
            return
        return 100 * pk_array + Plus

    @staticmethod
    def picket_to_string(distance):
        "Строковое представление пикета"
        pk_array, Plus = Picketazh.transform_dist_to_picket(distance)
        S = go.opts.translate("ПК{:d}+{:02.2f}").format(pk_array, Plus)
        if S[-5:] == "+0.00":
            S = S[:-5]
        elif S[-3:] == ".00":
            S = S[:-3]
        return S

    @staticmethod
    def string_to_picket(S):
        elements = S.split("+")
        if len(elements) != 2:
            return
        try:
            pk_array = int(elements[0])
        except:
            return None
        try:
            Plus = float(elements[1])
        except:
            return None
        return 100 * pk_array + Plus


class Calculations:
    @staticmethod
    def interpolate(x1, y1, x2, y2, x):
        """Линейная интерполяция"""
        try:
            return (x - x1) * (y2 - y1) / (x2 - x1) + y1
        except ZeroDivisionError:
            return None

    @staticmethod
    def calc_precision(X):
        """Расчет требуемого количества знаков после запятой"""
        dX = np.abs(X[1:] - X[:-1])
        dx = np.min(dX)
        if dx <= 1e-6:
            S = "{value:.{prec}g}".format(value=X[0], prec=6)
            d = decimal.Decimal(S)
            digits = np.abs(d.as_tuple().exponent)
            return min(digits, 6)
        k = 0
        while np.round(dx) == 0:
            k += 1
            dx *= 10
        return min(k, 6)


    @staticmethod
    def set_hist_levels(data):
        contrast = 100 - go.opts.radargramm.contrast
        median_amplitude = np.median(data)
        min_amplitude = np.min(data)
        max_amplitude = np.max(data)

        minCount = np.count_nonzero(data[data == min_amplitude])
        maxCount = np.count_nonzero(data[data == max_amplitude])
        if minCount + maxCount > 0.5 * (data.shape[0] * data.shape[1]):
            level1 = min_amplitude
            level2 = max_amplitude
        else:
            middleA = (min_amplitude + max_amplitude) / 2

            a = abs(min_amplitude - median_amplitude)
            b = abs(max_amplitude - median_amplitude)
            c = abs(middleA - median_amplitude)

            d1 = median_amplitude - min_amplitude
            d2 = max_amplitude - median_amplitude
            if min(a, b) > c:
                # mean находится ближе к середине (middleA), чем к границе (min_amplitude или max_amplitude)
                d1 = d2 = min(d1, d2)

            d1_new = d1 * contrast / 100
            d2_new = d2 * contrast / 100
            level1 = median_amplitude - d1_new
            level2 = median_amplitude + d2_new

            if min_amplitude >= 0:
                level1 = 0
        return level1, level2


class DateTransforms:
    @staticmethod
    def date_time_to_interval(date_time):
        null_date = datetime.datetime(1601, 1, 1, 0, 0, 0)
        try:
            date_time = date_time.to_pydatetime()
        except AttributeError:
            pass
        diff = date_time - null_date
        return diff.total_seconds() * 10000000

    @staticmethod
    def get_date_time(interval, is_gps_hours_delta=False):
        if (interval is None) or (interval == 0):
            return None
        try:
            FILETIME_null_date = datetime.datetime(1601, 1, 1, 0, 0, 0)
            if is_gps_hours_delta:
                gps_hours_delta = go.opts.processing.gps_hours_delta
                time = FILETIME_null_date + datetime.timedelta(microseconds=interval / 10) + datetime.timedelta(
                    hours=gps_hours_delta)
            else:
                time = FILETIME_null_date + datetime.timedelta(microseconds=interval / 10)
            date = pd.to_datetime(time)
            return date
        except (ValueError, OverflowError):
            return None

    @staticmethod
    def parser_date(date_time, date_time_parser_format):
        date_time = datetime.datetime.strptime(date_time, date_time_parser_format)
        if (date_time.year == 1900) and (go.project.radargram.time_collecting is not None):
            new_date = DateTransforms.get_date_time(go.project.radargram.time_collecting[0])
            date_time = pd.Timestamp(year=new_date.year, month=new_date.month, day=new_date.day,
                                    hour=date_time.hour, minute=date_time.minute, second=date_time.second,
                                    microsecond=date_time.microsecond)
        return date_time


class CoordTransformation:
    @staticmethod
    def geographical_to_cartesian(latitude, longitude):
        """Преобразование географических координат в плоские"""
        X = np.array([utm.from_latlon(xy[0], xy[1])[0] for xy in zip(latitude, longitude)])
        Y = np.array([utm.from_latlon(xy[0], xy[1])[1] for xy in zip(latitude, longitude)])
        try:
            _, _, zone_number, zone_letter = utm.from_latlon(latitude[0], longitude[0])
        except utm.error.OutOfRangeError:
            print("utm.error.OutOfRangeError")
            return False, None
        return True, [X, Y, zone_number, zone_letter]

    @staticmethod
    def cartesian_to_geographical(X, Y, zone_number, zone_letter):
        """Преобразование плоских координат в географические"""
        latitude = [utm.to_latlon(xy[0], xy[1], zone_number=zone_number, zone_letter=zone_letter, strict=False)[0] for xy in zip(X, Y)]
        longitude = [utm.to_latlon(xy[0], xy[1], zone_number=zone_number, zone_letter=zone_letter, strict=False)[1] for xy in zip(X, Y)]
        return latitude, longitude


def in_directory(file, directory):
    directory = os.path.join(os.path.realpath(directory), '')
    file = os.path.realpath(file)
    return os.path.commonprefix([file, directory]) == directory


def file_name_to_hash(file_name):
    hash_code = hashlib.md5(file_name.encode()).hexdigest()
    return os.path.splitext(file_name)[0] + hash_code + ".tmp"


def pick_file_data_for_terrazond(file_name):
    try:
        basename = os.path.basename(file_name)
        index = basename.rfind("_")
        if index == -1:
            return None

        if index == len(basename)-8:
            part_number = int(basename[index+1:-4])
            basename = basename[:index]
        else:
            part_number = 0
            basename = basename[:-4]

        index = basename.rfind("_")
        if index == -1:
            return None
        route_name = basename[:index]
        canal_string = basename[index+1:]
        canal_number = int(canal_string[6:])
        return route_name, canal_number, part_number
    except (ValueError, IndexError):
        return None

