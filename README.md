# GaitSet

A flexible, effective and fast network for cross-view gait recognition.
It consistent with the results in [GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition](https://arxiv.org/abs/1811.06186)

We arrived **Rank@1=95.0%** on [CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp) 
and  **Rank@1=87.1%** on [OU-MVLP](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html).

## Prerequisites

- Python 3.6
- PyTorch 0.4+
- GPU

## Getting started
### Installation

- (Not necessary) Install [Anaconda3](https://www.anaconda.com/download/)
- Install [CUDA 9.0](https://developer.nvidia.com/cuda-90-download-archive)
- install [cuDNN7.0](https://developer.nvidia.com/cudnn)
- Install [PyTorch](http://pytorch.org/)

Noted that our code is tested based on PyTorch 0.4

### Dataset & Preparation
Download [CASIA-B Dataset](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)

**ATTENTION**
- Organize the directory as: 
`your_dataset_path/resolutions/dataset_names/subject_ids/walking_conditions/views`.
E.g. `gaitdata/64/CASIA-B/001/nm-01/000/`. (We will update the code to be more compatible.)
- You should cut and align the raw silhouette by yourself. Our experiments use the align method in 
[this paper](https://ipsjcva.springeropen.com/articles/10.1186/s41074-018-0039-6).

Futhermore, you also can test our code on [OU-MVLP Dataset](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html).
The number of channels and the training batchsize is slightly different for this dataset.
For more detail, please refer to [our paper](https://arxiv.org/abs/1811.06186).

### Configuration 

In `config.py`, you might want to change the following settings:
- `WORK_PATH` path to save/load checkpoints
- `CUDA_VISIBLE_DEVICES` indices of GPUs
- `dataset_path` (necessary) root path of the dataset 
(for the above example, it is "gaitdata")

### Train
Train a model by
```bash
python train.py
```

### Test & Evaluation
Use trained model to extract feature by
```bash
python test.py
```
- `--iter` iteration of the checkpoint to load
- `--batch_size` batch size of the parallel test

It will output Rank@1 of all three walking conditions. 
Note that the test is **parallelizable**. 
To conduct a faster evaluation, you could use `--batch_size` to change the batch size for test.
