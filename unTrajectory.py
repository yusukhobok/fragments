import io

import utm
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import unGlobal
from unGlobal import go
from unUtils import DateTransforms, Geometry, CoordTransformation


class Trajectory:
    """Класс GPS-траектории"""

    def __init__(self, parent, gps_trajectory=None, is_geographical=False, is_by_time=True):
        self._parent = parent
        self.gps_trajectory = gps_trajectory  # данные о траектории GPS ['X', 'Y', 'Z', 'date_time'/'trace_number']
        self.is_geographical = is_geographical  # являются ли координаты географическими (т.е. хранят в себе ширину и долготу)
        self.is_by_time = is_by_time  # синхронизирована ли траектория с временем снятия георадарных трасс

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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_parent"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.parent = None

    def from_coords(self, is_geographical, is_drop_duplicates=True):
        """подгрузка траектории из координат"""
        if self.parent.radargram.time_collecting is None:
            return None
        trace_coordinate_array = self.parent.radargram.trace_coordinate_array
        self.gps_trajectory = pd.DataFrame({"X": trace_coordinate_array["X"].values, "Y": trace_coordinate_array["Y"].values, "Z": trace_coordinate_array["Z"].values,
                                           "date_time": [DateTransforms.get_date_time(interval, is_gps_hours_delta=True) for interval in self.parent.radargram.time_collecting]})
        if is_drop_duplicates:
            self.gps_trajectory = self.gps_trajectory.drop_duplicates(subset=["X", "Y"]).reset_index(drop=True)
        self.is_geographical = is_geographical
        self.is_by_time = True

    def _shift_trajectory_longitunial(self, X, Y, gps_trajectory, offset_longitunial):
        segments_distance = np.sqrt((X[1:] - X[:-1]) ** 2 + (Y[1:] - Y[:-1]) ** 2)
        sum_distance = np.cumsum(segments_distance)
        sum_distance = np.insert(sum_distance, 0, 0)
        sum_distance, unique_indices = np.unique(sum_distance, return_index=True)
        X = X[unique_indices]
        Y = Y[unique_indices]

        gps_trajectory = pd.DataFrame(gps_trajectory.iloc[unique_indices])
        new_sum_distance = sum_distance + offset_longitunial
        fX = interp1d(sum_distance, X, kind='linear', fill_value='extrapolate')
        X = fX(new_sum_distance)
        fY = interp1d(sum_distance, Y, kind='linear', fill_value='extrapolate')
        Y = fY(new_sum_distance)
        return X, Y, gps_trajectory

    def _shift_trajectory_normal(self, X, Y, offset_normal):
        x1p = None
        y1p = None
        x2p = None
        y2p = None
        NewX = []
        NewY = []
        for i in range(len(X) - 1):
            x1p_last, y1p_last, x2p_last, y2p_last = x1p, y1p, x2p, y2p

            x1 = X[i]
            x2 = X[i + 1]
            y1 = Y[i]
            y2 = Y[i + 1]
            x1p, y1p, x2p, y2p = Geometry.offset_segment(x1, y1, x2, y2, offset_normal)

            if i == 0:
                if np.isnan(x1p) or np.isnan(y1p):
                    x1p = y1p = 0
                else:
                    NewX.append(x1p)
                    NewY.append(y1p)
            else:
                if np.isnan(x1p) or np.isnan(y1p) or np.isnan(x2p) or np.isnan(y2p):
                    x1p, y1p, x2p, y2p = x1p_last, y1p_last, x2p_last, y2p_last
                    NewX.append(NewX[-1])
                    NewY.append(NewY[-1])
                else:
                    NewX.append((x1p + x2p_last) / 2)
                    NewY.append((y1p + y2p_last) / 2)
            if i == len(X) - 2:
                if np.isnan(x2p) or np.isnan(y2p):
                    NewX.append(NewX[-1])
                    NewY.append(NewY[-1])
                else:
                    NewX.append(x2p)
                    NewY.append(y2p)
        return NewX, NewY

    def shift_trajectory(self, gps_trajectory, offset_normal=0, offset_longitunial=0):
        """сдвижка траектории в продольном направлении (offset_longitunial, м) 
        и в поперечном направлении (offset_normal, м)"""
        if gps_trajectory is None:
            return False, gps_trajectory
        if (offset_normal == 0) and (offset_longitunial == 0):
            return True, gps_trajectory

        if self.is_geographical:
            longitude = gps_trajectory["X"].values
            latitude = gps_trajectory["Y"].values
            res, data = CoordTransformation.geographical_to_cartesian(latitude, longitude)
            if not res:
                return False, gps_trajectory
            X, Y, zone_number, zone_letter = data
        else:
            X = gps_trajectory["X"].values
            Y = gps_trajectory["Y"].values

        if offset_longitunial != 0:
            X, Y, gps_trajectory = self._shift_trajectory_longitunial(X, Y, gps_trajectory, offset_longitunial)
        if offset_normal != 0:
            NewX, NewY = self._shift_trajectory_normal(X, Y, offset_normal)
            if self.is_geographical:
                new_latitude, new_longitude = CoordTransformation.cartesian_to_geographical(NewX, NewY, zone_number,
                                                                                            zone_letter)
                if (len(new_latitude) != len(latitude)) or (len(new_longitude) != len(longitude)):
                    print(go.opts.translate("Не удалось откорректировать GPS-траекторию"))
                    return False, gps_trajectory

                gps_trajectory["X"] = new_longitude
                gps_trajectory["Y"] = new_latitude
            else:
                gps_trajectory["X"] = NewX
                gps_trajectory["Y"] = NewY
        else:
            gps_trajectory["X"] = X
            gps_trajectory["Y"] = Y
        return True, gps_trajectory

    def correct_coords(self, offset_normal=0, offset_longitunial=0):
        """корректировка координат за счет сдвижки траектории"""
        result, gps_trajectory = self.shift_trajectory(self.gps_trajectory, offset_normal, offset_longitunial)
        if result:
            self.gps_trajectory = gps_trajectory
        return result

    def import_gps_trajectory(self, file_name, offset_normal=0, offset_longitunial=0):
        """Импорт траектории из формата Картскан"""
        self.is_geographical = True
        self.is_by_time = True

        f = open(file_name)
        gps_trajectory = pd.read_csv(f, sep='\t', encoding='cp1251', names=["date", "time", "Y", "N_or_S", "X", "E_or_W", "Z"],
                                     index_col=None)
        print(gps_trajectory)
        gps_trajectory = gps_trajectory[gps_trajectory["date"] != 'M']
        gps_trajectory["date_time"] = gps_trajectory["date"] + " " + gps_trajectory["time"]
        gps_trajectory["date_time"] = gps_trajectory["date_time"].apply(DateTransforms.parser_date, args=("%Y:%m:%d %H:%M:%S.%f", ))
        gps_trajectory["X"] = gps_trajectory["X"].astype("float64")
        gps_trajectory["Y"] = gps_trajectory["Y"].astype("float64")
        gps_trajectory.loc[gps_trajectory["N_or_S"] == "S", "Y"] = - gps_trajectory["Y"]
        gps_trajectory.loc[gps_trajectory["E_or_W"] == "W", "X"] = - gps_trajectory["X"]
        gps_trajectory = gps_trajectory.drop(["date", "time", "N_or_S", "E_or_W"], axis=1)
        gps_trajectory["Z"] = gps_trajectory["Z"].fillna(0)

        result, gps_trajectory_shifted = self.shift_trajectory(gps_trajectory, offset_normal, offset_longitunial)
        if result:
            gps_trajectory = gps_trajectory_shifted
        gps_trajectory = self.cut_trajectory(gps_trajectory)

        if gps_trajectory.shape[0] < 2:
            return None
        if gps_trajectory is None:
            return None
        self.gps_trajectory = gps_trajectory
        return True

    def import_equalized_gps_trajectory(self, file_name, offset_normal=0, offset_longitunial=0):
        """Импорт уравненной GPS-траектории"""
        self.is_geographical = True
        self.is_by_time = True

        f = open(file_name)
        S = f.readlines()
        S = S[20:-4]
        S = [el.split() for el in S]

        columns = ["Station", "UTCTime", "Easting", "Northing", "Latitude_Deg", "Latitude_Min", "Latitude_Sec",
                   "Longitude_Deg", "Longitude_Min", "Longitude_Sec", "height", "date", "time", "Week"]
        gps_trajectory = pd.DataFrame(S, columns=columns)

        gps_trajectory["Y"] = gps_trajectory["Latitude_Deg"].astype("float64") + gps_trajectory["Latitude_Min"].astype("float64")/60 + gps_trajectory["Latitude_Sec"].astype("float64")/3600
        gps_trajectory["X"] = gps_trajectory["Longitude_Deg"].astype("float64") + gps_trajectory["Longitude_Min"].astype("float64")/60 + gps_trajectory["Longitude_Sec"].astype("float64")/3600
        gps_trajectory["Y"] = gps_trajectory["Y"].astype("float64")
        gps_trajectory["X"] = gps_trajectory["X"].astype("float64")
        gps_trajectory["date_time"] = gps_trajectory["date"] + " " + gps_trajectory["time"]
        gps_trajectory["date_time"] = gps_trajectory["date_time"].apply(DateTransforms.parser_date, args=("%m/%d/%Y %H:%M:%S",))
        gps_trajectory = gps_trajectory.drop(["Latitude_Deg", "Latitude_Min", "Latitude_Sec",
                                            "Longitude_Deg", "Longitude_Min", "Longitude_Sec",
                                            "Easting", "Northing", "Week", "Station", "height",
                                            "date", "time", "UTCTime"], axis=1)
        gps_trajectory["date_time"] = gps_trajectory["date_time"] - pd.to_timedelta(go.opts.processing.gps_hours_delta, 'h')
        gps_trajectory["Z"] = 0

        result, gps_trajectory_shifted = self.shift_trajectory(gps_trajectory, offset_normal, offset_longitunial)
        if result:
            gps_trajectory = gps_trajectory_shifted
        gps_trajectory = self.cut_trajectory(gps_trajectory)

        if gps_trajectory.shape[0] < 2:
            return None
        if gps_trajectory is None:
            return None
        self.gps_trajectory = gps_trajectory
        return True

    def import_equalized_gps_trajectory_from_dataframe(self, df):
        self.gps_trajectory = df.copy()

    def import_gps_trajectory_log_geoscan(self, file_name, offset_normal=0, offset_longitunial=0):
        """Импорт траектории из формата log (GeoScan)"""
        self.is_geographical = True
        self.is_by_time = True

        f = io.open(file_name, 'r', encoding='utf-16-le')
        lines = f.readlines()
        del (lines[-1])
        lines = [line[:-1] for line in lines]
        raw_data = [line.split(" ") for line in lines]
        raw_data = [[el for el in line if el != ''] for line in raw_data]

        raw_data = pd.DataFrame(raw_data).values
        if raw_data.shape[0] < 2:
            return None

        time_collecting = raw_data[:,0].astype("uint64")
        X = raw_data[:,2].astype("float64") + raw_data[:,3].astype("float64")/60
        E_or_W = raw_data[:,4]
        Y = raw_data[:, 5].astype("float64") + raw_data[:, 6].astype("float64")/60
        N_or_S = raw_data[:, 7]
        Z = raw_data[:, 8].astype("float64")

        gps_trajectory = pd.DataFrame({"time_collecting": time_collecting, "X": X, "N_or_S": N_or_S, "Y": Y, "E_or_W": E_or_W, "Z": Z})
        gps_trajectory.loc[gps_trajectory["N_or_S"] == "S", "Y"] = - gps_trajectory["Y"]
        gps_trajectory.loc[gps_trajectory["E_or_W"] == "W", "X"] = - gps_trajectory["X"]
        gps_trajectory["date_time"] = gps_trajectory["time_collecting"].apply(DateTransforms.get_date_time, args=(True,))
        gps_trajectory = gps_trajectory.drop(["time_collecting", "N_or_S", "E_or_W"], axis=1)
        result, gps_trajectory_shifted = self.shift_trajectory(gps_trajectory, offset_normal, offset_longitunial)
        if result:
            gps_trajectory = gps_trajectory_shifted
        gps_trajectory = self.cut_trajectory(gps_trajectory)

        if gps_trajectory is None:
            return None
        if gps_trajectory.shape[0] < 2:
            return None
        self.gps_trajectory = gps_trajectory
        return True

    def import_gps_trajectory_arbitrary(self, trajectory, is_geographical, is_by_time=True, offset_normal=0, offset_longitunial=0):
        """Импорт траектории в произвольном формате"""
        try:
            result, gps_trajectory_shifted = self.shift_trajectory(trajectory, offset_normal, offset_longitunial)
            if result:
                trajectory = gps_trajectory_shifted
            gps_trajectory = self.cut_trajectory(trajectory, is_by_time)
            if gps_trajectory is None:
                return None
            if gps_trajectory.shape[0] < 2:
                return None
            self.gps_trajectory = gps_trajectory
            self.is_geographical = is_geographical
            self.is_by_time = is_by_time
            # print(self.gps_trajectory)
            return True
        except:
            print("Ошибка импорта траектории")
            return None

    def cut_trajectory(self, gps_trajectory, is_by_time=True):
        """Прореживание GPS-траектории"""
        if is_by_time:
            if self.parent.radargram.time_collecting is None:
                return gps_trajectory
            try:
                date_collecting = np.array([DateTransforms.get_date_time(interval, is_gps_hours_delta=True) for interval in self.parent.radargram.time_collecting])
            except OverflowError:
                return None
            if (date_collecting[0] is None) or (date_collecting[-1] is None):
                return None
            gps_trajectory = gps_trajectory[(gps_trajectory["date_time"] >= date_collecting[0]) & (gps_trajectory["date_time"] <= date_collecting[-1])]
            return gps_trajectory
        else:
            return gps_trajectory

    def reflect(self):
        """Отражение траектории"""
        if self.gps_trajectory is None:
            return
        self.gps_trajectory["X"] = self.gps_trajectory["X"].values[::-1]
        self.gps_trajectory["Y"] = self.gps_trajectory["Y"].values[::-1]
        self.gps_trajectory["Z"] = self.gps_trajectory["Z"].values[::-1]
        if "date_time" in self.gps_trajectory:
            self.gps_trajectory["date_time"] = self.gps_trajectory["date_time"].values[::-1]

    def find_first_and_last_coords(self):
        if self.gps_trajectory is None:
            return None
        return self.gps_trajectory[["X", "Y"]].values[[0, -1]]
