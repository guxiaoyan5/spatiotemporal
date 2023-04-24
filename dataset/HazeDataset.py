import os
from collections import OrderedDict
from datetime import datetime

import arrow
import metpy.calc as mpcalc
import numpy as np
import torch
from bresenham import bresenham
from geopy.distance import geodesic
from metpy.units import units
from scipy.spatial import distance
from torch.utils import data
from torch_geometric.utils import to_dense_adj, dense_to_sparse

mete_var = ['100m_u_component_of_wind',
            '100m_v_component_of_wind',
            '2m_dewpoint_temperature',
            '2m_temperature',
            'boundary_layer_height',
            'k_index',
            'relative_humidity+950',
            'relative_humidity+975',
            'specific_humidity+950',
            'surface_pressure',
            'temperature+925',
            'temperature+950',
            'total_precipitation',
            'u_component_of_wind+950',
            'v_component_of_wind+950',
            'vertical_velocity+950',
            'vorticity+950']


class HazeData(data.Dataset):

    def __init__(self, config, flag='Train', ):
        if flag == 'Train':
            start_time_str = 'train_start'
            end_time_str = 'train_end'
        elif flag == 'Val':
            start_time_str = 'val_start'
            end_time_str = 'val_end'
        elif flag == 'Test':
            start_time_str = 'test_start'
            end_time_str = 'test_end'
        else:
            raise Exception('Wrong Flag!')
        self.config = config
        self.start_time = self._get_time(config['dataset'][start_time_str])
        self.end_time = self._get_time(config['dataset'][end_time_str])

        self.train_start = self._get_time(config['dataset']['train_start'])
        self.train_end = self._get_time(config['dataset']['train_end'])
        self.data_start = self._get_time(config['dataset']['data_start'])
        self.data_end = self._get_time(config['dataset']['data_end'])
        self.knowAir_fp = config["dataset"]['knowAir_fp']
        self.pm25, self.feature = self._load_npy()
        self.time_arr, self.time_arrow = self._gen_time_arr()
        self.pm25, self.feature, self.time_arr, self.time_arrow = self._process_time(self.data_start, self.data_end)
        self._process_feature()
        self._calc_mean_std()
        self.pm25, self.feature, self.time_arr, self.time_arrow = self._process_time(self.start_time, self.end_time)
        self.feature = np.float32(self.feature)
        self.pm25 = np.float32(self.pm25)
        seq_len = config['experiments']['hist_len'] + config['experiments']['pred_len']
        self._add_time_dim(seq_len)
        self._norm()

    def _norm(self):
        self.feature = (self.feature - self.feature_mean) / self.feature_std
        self.pm25 = (self.pm25 - self.pm25_mean) / self.pm25_std

    def _add_time_dim(self, seq_len):

        def _add_t(arr):
            t_len = arr.shape[0]
            assert t_len > seq_len
            arr_ts = []
            for i in range(seq_len, t_len):
                arr_t = arr[i - seq_len:i]
                arr_ts.append(arr_t)
            arr_ts = np.stack(arr_ts, axis=0)
            return arr_ts

        self.pm25 = _add_t(self.pm25)
        self.feature = _add_t(self.feature)
        self.time_arr = _add_t(self.time_arr)

    def _calc_mean_std(self):
        start_idx = self._get_idx(self.train_start)
        end_idx = self._get_idx(self.train_end)
        self.feature_mean = self.feature[start_idx: end_idx + 1, :].mean(axis=(0, 1))
        self.feature_std = self.feature[start_idx: end_idx + 1, :].std(axis=(0, 1))
        self.wind_mean = self.feature_mean[-2:]
        self.wind_std = self.feature_std[-2:]
        self.pm25_mean = self.pm25[start_idx: end_idx + 1, :].mean()
        self.pm25_std = self.pm25[start_idx: end_idx + 1, :].std()

    def _process_feature(self):
        mete_use = self.config['experiments']['mete_use']
        mete_idx = [mete_var.index(var) for var in mete_use]
        self.feature = self.feature[:, :, mete_idx]

        u = self.feature[:, :, -2] * units.meter / units.second
        v = self.feature[:, :, -1] * units.meter / units.second
        speed = 3.6 * mpcalc.wind_speed(u, v).magnitude
        direc = mpcalc.wind_direction(u, v).magnitude

        h_arr = []
        w_arr = []
        for i in self.time_arrow:
            h_arr.append(i.hour)
            w_arr.append(i.isoweekday())
        h_arr = np.stack(h_arr, axis=-1)
        w_arr = np.stack(w_arr, axis=-1)
        h_arr = np.repeat(h_arr[:, None], self.config["dataset"]["node_num"], axis=1)
        w_arr = np.repeat(w_arr[:, None], self.config["dataset"]["node_num"], axis=1)

        self.feature = np.concatenate(
            [self.feature, h_arr[:, :, None], w_arr[:, :, None], speed[:, :, None], direc[:, :, None]], axis=-1)

    def _process_time(self, start_time, end_time):
        start_idx = self._get_idx(start_time)
        end_idx = self._get_idx(end_time)
        pm25 = self.pm25[start_idx: end_idx + 1, :]
        feature = self.feature[start_idx: end_idx + 1, :]
        time_arr = self.time_arr[start_idx: end_idx + 1]
        time_arrow = self.time_arrow[start_idx: end_idx + 1]
        return pm25, feature, time_arr, time_arrow

    def _gen_time_arr(self):
        time_arr = []
        time_arrows = []
        for time_arrow in arrow.Arrow.interval('hour', self.data_start, self.data_end.shift(hours=+3), 3):
            time_arrows.append(time_arrow[0])
            time_arr.append(time_arrow[0].timestamp())
        time_arr = np.stack(time_arr, axis=-1)
        return time_arr, time_arrows

    def _load_npy(self):
        knowAir = np.load(self.knowAir_fp)
        feature = knowAir[:, :, :-1]
        pm25 = knowAir[:, :, -1:]
        return pm25, feature

    def _get_idx(self, t):
        t0 = self.data_start
        return int((t.timestamp() - t0.timestamp()) / (60 * 60 * 3))

    @classmethod
    def _get_time(cls, time_yaml):
        arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1])
        return arrow_time

    def __len__(self):
        return len(self.pm25)

    def __getitem__(self, index):
        return self.pm25[index, :self.config['experiments']['hist_len']], self.feature[index], self.time_arr[
            index], self.pm25[index, self.config['experiments']['hist_len']:]


class Graph:
    def __init__(self, altitude_fp="data/KnowAir/altitude.npy", city_fp="data/KnowAir/city.txt"):
        self.dist_thres = 3
        self.alti_thres = 1200
        self.use_altitude = True
        self.altitude_fp = os.path.join(altitude_fp)
        self.city_fp = os.path.join(city_fp)

        self.altitude = self._load_altitude()
        self.nodes = self._gen_nodes()
        self.node_attr = self._add_node_attr()
        self.node_num = len(self.nodes)
        self.edge_index, self.edge_attr = self._gen_edges()
        if self.use_altitude:
            self._update_edges()
        self.edge_num = self.edge_index.shape[1]
        self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]

    def _load_altitude(self):
        assert os.path.isfile(self.altitude_fp)
        altitude = np.load(self.altitude_fp)
        return altitude

    def _lonlat2xy(self, lon, lat):

        lon_l = 100.0
        lon_r = 128.0
        lat_u = 48.0
        lat_d = 16.0
        res = 0.05

        x = np.int64(np.round((lon - lon_l - res / 2) / res))
        y = np.int64(np.round((lat_u + res / 2 - lat) / res))
        return x, y

    def _gen_nodes(self):
        nodes = OrderedDict()
        with open(self.city_fp, 'r') as f:
            for line in f:
                idx, city, lon, lat = line.rstrip('\n').split(' ')
                idx = int(idx)
                lon, lat = float(lon), float(lat)
                x, y = self._lonlat2xy(lon, lat)
                altitude = self.altitude[y, x]
                nodes.update({idx: {'city': city, 'altitude': altitude, 'lon': lon, 'lat': lat}})
        return nodes

    def _add_node_attr(self):
        node_attr = []
        altitude_arr = []
        for i in self.nodes:
            altitude = self.nodes[i]['altitude']
            altitude_arr.append(altitude)
        altitude_arr = np.stack(altitude_arr)
        node_attr = np.stack([altitude_arr], axis=-1)
        return node_attr

    def traverse_graph(self):
        lons = []
        lats = []
        citys = []
        idx = []
        for i in self.nodes:
            idx.append(i)
            city = self.nodes[i]['city']
            lon, lat = self.nodes[i]['lon'], self.nodes[i]['lat']
            lons.append(lon)
            lats.append(lat)
            citys.append(city)
        return idx, citys, lons, lats

    def gen_lines(self):

        lines = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            lines.append(([src_lon, dest_lon], [src_lat, dest_lat]))

        return lines

    def _gen_edges(self):
        coords = []
        lonlat = {}
        for i in self.nodes:
            coords.append([self.nodes[i]['lon'], self.nodes[i]['lat']])
        dist = distance.cdist(coords, coords, 'euclidean')
        adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        adj[dist <= self.dist_thres] = 1
        assert adj.shape == dist.shape
        dist = dist * adj
        edge_index, dist = dense_to_sparse(torch.tensor(dist))
        edge_index, dist = edge_index.numpy(), dist.numpy()

        direc_arr = []
        dist_kilometer = []
        for i in range(edge_index.shape[1]):
            src, dest = edge_index[0, i], edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_location = (src_lat, src_lon)
            dest_location = (dest_lat, dest_lon)
            dist_km = geodesic(src_location, dest_location).kilometers
            v, u = src_lat - dest_lat, src_lon - dest_lon

            u = u * units.meter / units.second
            v = v * units.meter / units.second
            direc = mpcalc.wind_direction(u, v)._magnitude

            direc_arr.append(direc)
            dist_kilometer.append(dist_km)

        direc_arr = np.stack(direc_arr)
        dist_arr = np.stack(dist_kilometer)
        attr = np.stack([dist_arr, direc_arr], axis=-1)

        return edge_index, attr

    def _update_edges(self):
        edge_index = []
        edge_attr = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_x, src_y = self._lonlat2xy(src_lon, src_lat)
            dest_x, dest_y = self._lonlat2xy(dest_lon, dest_lat)
            points = np.asarray(list(bresenham(src_y, src_x, dest_y, dest_x))).transpose((1, 0))
            altitude_points = self.altitude[points[0], points[1]]
            altitude_src = self.altitude[src_y, src_x]
            altitude_dest = self.altitude[dest_y, dest_x]
            if np.sum(altitude_points - altitude_src > self.alti_thres) < 3 and \
                    np.sum(altitude_points - altitude_dest > self.alti_thres) < 3:
                edge_index.append(self.edge_index[:, i])
                edge_attr.append(self.edge_attr[i])

        self.edge_index = np.stack(edge_index, axis=1)
        self.edge_attr = np.stack(edge_attr, axis=0)


if __name__ == '__main__':
    graph = Graph(altitude_fp="../data/KnowAir/altitude.npy", city_fp="../data/KnowAir/city.txt")
