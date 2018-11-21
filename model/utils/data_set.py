import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import xarray as xr


class DataSet(tordata.Dataset):
    def __init__(self, seq_dir, label, seq_type, view):
        self.seq_dir = seq_dir
        self.view = view
        self.seq_type = seq_type
        self.label = label
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size

        self.label_set = set(self.label)
        self.seq_type_set = set(self.seq_type)
        self.view_set = set(self.view)
        _ = np.zeros((len(self.label_set),
                      len(self.seq_type_set),
                      len(self.view_set))).astype('int')
        _ -= 1
        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set)),
                    'seq_type': sorted(list(self.seq_type_set)),
                    'view': sorted(list(self.view_set))},
            dims=['label', 'seq_type', 'view'])

        for i in range(self.data_size):
            _label = self.label[i]
            _seq_type = self.seq_type[i]
            _view = self.view[i]
            self.index_dict.loc[_label, _seq_type, _view] = i

        self.selector = dict()

    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, path):
        file_path = osp.join(path, os.listdir(path)[0])
        seq = pickle.load(open(file_path, 'rb'))
        data_type = seq.attrs['data_type']
        data_name = seq.name
        if data_name in self.selector:
            seq = seq.loc.__getitem__(tuple(self.selector[data_name]))
        if data_type == 'img':
            seq = seq.astype('float32')
            seq /= 255.0
            seq = seq[:, :, 10:54]
        return seq

    def __getitem__(self, index):
        # pose sequence sampling
        if self.data[index] == None:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            self.data[index] = data
            self.frame_set[index] = frame_set
        return self.data[index], self.frame_set[index], self.view[
            index], self.seq_type[index], self.label[index],

    def __len__(self):
        return len(self.label)
