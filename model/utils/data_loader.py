import os
import os.path as osp

import numpy as np

from .data_set import DataSet


def load_data(dataset_path, resolution, dataset, pid_num, pid_shuffle, feature='silhouettes'):
    data_dir = osp.join(dataset_path, resolution, dataset, feature)
    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()
    dirpath_set = set()

    for label_dir in os.listdir(data_dir):
        label_dir_path=osp.join(data_dir,label_dir)
        for seq_type_dir in os.listdir(label_dir_path):
            seq_type_dir_path=osp.join(label_dir_path,seq_type_dir)
            for view_dir in os.listdir(seq_type_dir_path):
                view_dir_path=osp.join(seq_type_dir_path,view_dir)
                for frame in os.listdir(view_dir_path):
                    if frame[-3:] == 'jpg':
                        dirpath_set.add(view_dir_path)
                        break


    dirpath_set=sorted(list(dirpath_set))
    for dirpath in dirpath_set:
        _label, _seq_type, _view = dirpath.split('/')[-3:]
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        if _label == '005':
            continue
        _seq_dir = list()
        _path = osp.join(data_dir, _label, _seq_type, _view)
        if osp.isdir(_path) and len(os.listdir(_path)) != 0:
            _seq_dir.append(osp.abspath(_path))

        if len(_seq_dir) == 1:
            seq_dir.append(_seq_dir)
            label.append(_label)
            seq_type.append(_seq_type)
            view.append(_view)

    pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
        dataset, pid_num, pid_shuffle))
    if not osp.exists(pid_fname):
        pid_list = sorted(list(set(label)))
        if pid_shuffle:
            np.random.shuffle(pid_list)
        pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
        os.makedirs('partition', exist_ok=True)
        np.save(pid_fname, pid_list)

    pid_list = np.load(pid_fname)
    train_list = pid_list[0]
    test_list = pid_list[1]

    train_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in train_list],
        [label[i] for i, l in enumerate(label) if l in train_list],
        [seq_type[i] for i, l in enumerate(label) if l in train_list],
        [view[i] for i, l in enumerate(label)
         if l in train_list])
    test_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in test_list],
        [label[i] for i, l in enumerate(label) if l in test_list],
        [seq_type[i] for i, l in enumerate(label) if l in test_list],
        [view[i] for i, l in enumerate(label)
         if l in test_list])

    return train_source, test_source
