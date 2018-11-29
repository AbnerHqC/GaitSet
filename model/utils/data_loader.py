import os
import os.path as osp

import numpy as np

from .data_set import DataSet


def load_data(dataset_path, resolution, dataset, pid_num, pid_shuffle, cache=True):
    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()

    for _label in sorted(list(os.listdir(dataset_path))):
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        if dataset == 'CASIA-B' and _label == '005':
            continue
        label_path = osp.join(dataset_path, _label)
        for _seq_type in sorted(list(os.listdir(label_path))):
            seq_type_path = osp.join(label_path, _seq_type)
            for _view in sorted(list(os.listdir(seq_type_path))):
                _seq_dir = osp.join(seq_type_path, _view)
                seqs = os.listdir(_seq_dir)
                if len(seqs) > 0:
                    seq_dir.append([_seq_dir])
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
         if l in train_list],
        cache, resolution)
    test_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in test_list],
        [label[i] for i, l in enumerate(label) if l in test_list],
        [seq_type[i] for i, l in enumerate(label) if l in test_list],
        [view[i] for i, l in enumerate(label)
         if l in test_list],
        cache, resolution)

    return train_source, test_source
