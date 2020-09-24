import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2
import xarray as xr

# extend torch.utils.data.Dataset,
# Build the dataset implementation for gait recognition,
# which include the parameters: label, type[NM, BG, CL], sequence id, view angle.
class DataSet(tordata.Dataset):
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution):
        self.seq_dir = seq_dir
        self.view = view
        self.seq_type = seq_type
        self.label = label
        self.cache = cache
        self.resolution = int(resolution)
        # cut_padding = 10
        self.cut_padding = int(float(resolution)/64*10)
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size

        # remove duplicated value
        self.label_set = set(self.label)
        self.seq_type_set = set(self.seq_type)
        self.view_set = set(self.view)
        # Return a new array of given shape and type, filled with zeros.
        _ = np.zeros((len(self.label_set),
                      len(self.seq_type_set),
                      len(self.view_set))).astype('int')
        _ -= 1
        # DataArray(values, Coordinates, dims)
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
            # Attribute for location based indexing. Only supports __getitem__,
            # and only when the key is a dict of the form {dim: labels}.
            self.index_dict.loc[_label, _seq_type, _view] = i

    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    # load the image to xarray
    def __loader__(self, path):
        return self.img2xarray(
            path)[:, :, self.cut_padding:-self.cut_padding].astype(
            'float32') / 255.0

    # overwrite the getitem function in the subclass
    def __getitem__(self, index):
        # pose sequence sampling
        if not self.cache:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
        elif self.data[index] is None:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            self.data[index] = data
            self.frame_set[index] = frame_set
        else:
            data = self.data[index]
            frame_set = self.frame_set[index]

        return data, frame_set, self.view[
            index], self.seq_type[index], self.label[index],

    # Image to Xarray
    def img2xarray(self, flie_path):
        imgs = sorted(list(os.listdir(flie_path)))
        # [:,:,0] Take the first dimension of the 3D data
        # [:,:,1] Take the second dimension of the 3D data
        # np.reshape the image data to [64, 64, -1]
        # When using a -1, the dimension corresponding to the -1
        # will be the product of the dimensions of the original array
        # divided by the product of the dimensions given to reshape
        # so as to maintain the same number of elements.
        frame_list = [np.reshape(
            cv2.imread(osp.join(flie_path, _img_path)),
            [self.resolution, self.resolution, -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(flie_path, _img_path))]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict

    # overwrite the len function in the subclass
    def __len__(self):
        return len(self.label)
