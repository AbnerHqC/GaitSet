import os
import os.path as osp

import numpy as np

from .data_set import DataSet


def load_data(dataset_path, resolution, dataset, features, pid_num, pid_shuffle):
    data_dir = osp.join(dataset_path, resolution, dataset)
    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()

    _feature_dir = osp.join(data_dir, features[0])
    for (dirpath, dirnames, filenames) in os.walk(_feature_dir):
        if len(dirnames) != 0:
            continue
        _label, _seq_type, _view = dirpath.split('/')[-3:]
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        if _label == '005':
            continue
        _seq_dir = list()
        for _feature in features:
            _path = osp.join(data_dir, _feature, _label, _seq_type, _view)
            if osp.isdir(_path) and len(os.listdir(_path)) != 0:
                _seq_dir.append(osp.abspath(_path))

        if len(_seq_dir) == len(features):
            seq_dir.append(_seq_dir)
            label.append(_label)
            seq_type.append(_seq_type)
            view.append(_view)

    pid_fname = osp.join('partition', '{}_{}_{}_{}.npy'.format(
        dataset, features, pid_num, pid_shuffle))
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
