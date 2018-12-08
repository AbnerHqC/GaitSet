# GaitSet

[GaitSet](https://arxiv.org/abs/1811.06186) is a **flexible**, **effective** and **fast** network for cross-view gait recognition.

#### Flexible 
The input of GaitSet is a set of silhouettes. 

- There are **NOT ANY constrains** on an input,
which means it can contain **any number** of **non-consecutive** silhouettes filmed under **different viewpoints**
with **different walking conditions**.

- As the input is a set, the **permutation** of the elements in the input
will **NOT change** the output at all.

#### Effective
It achieves **Rank@1=95.0%** on [CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp) 
and  **Rank@1=87.1%** on [OU-MVLP](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html),
excluding  identical-view cases.

#### Fast
With 8 NVIDIA 1080TI GPUs, It only takes **7 minutes** to conduct an evaluation on
[OU-MVLP](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html) which contains 133,780 sequences
and average 70 frames per sequence.

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

Noted that our code is tested based on [PyTorch 0.4](http://pytorch.org/)

### Dataset & Preparation
Download [CASIA-B Dataset](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)

**!!! ATTENTION !!! ATTENTION !!! ATTENTION !!!**
- Organize the directory as: 
`your_dataset_path/subject_ids/walking_conditions/views`.
E.g. `CASIA-B/001/nm-01/000/`.
- You need to cut and align the raw silhouettes in the dataset before you can use it.
Our experiments used the alignment method in 
[this paper](https://ipsjcva.springeropen.com/articles/10.1186/s41074-018-0039-6).
- The input sample **MUST be resized into 64x64**

Futhermore, you also can test our code on [OU-MVLP Dataset](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html).
The number of channels and the training batchsize is slightly different for this dataset.
For more detail, please refer to [our paper](https://arxiv.org/abs/1811.06186).

### Configuration 

In `config.py`, you might want to change the following settings:
- `dataset_path` **(NECESSARY)** root path of the dataset 
(for the above example, it is "gaitdata")
- `WORK_PATH` path to save/load checkpoints
- `CUDA_VISIBLE_DEVICES` indices of GPUs

### Train
Train a model by
```bash
python train.py
```
- `--cache` if set as TRUE all the training data will be loaded at once before the training start.
This will accelerate the training.
**Note that** if this arg is set as FALSE, samples will NOT be kept in the memory
even they have been used in the former iterations. #Default: TRUE

### Evaluation
Use trained model to extract feature by
```bash
python test.py
```
- `--iter` iteration of the checkpoint to load. #Default: 80000
- `--batch_size` batch size of the parallel test. #Default: 1
- `--cache` if set as TRUE all the test data will be loaded at once before the transforming start.
This might accelerate the testing. #Default: FALSE

It will output Rank@1 of all three walking conditions. 
Note that the test is **parallelizable**. 
To conduct a faster evaluation, you could use `--batch_size` to change the batch size for test.

## To Do List
- Pretreatment: The script for the pretreatment of CASIA-B dataset.
- Transformation: The script for transforming a set of silhouettes into a discriminative representation.


## Authors & Contributors
GaitSet is authored by
[Hanqing Chao](https://www.linkedin.com/in/hanqing-chao-9aa42412b/), 
[Yiwei He](https://www.linkedin.com/in/yiwei-he-4a6a6bbb/),
[Junping Zhang](http://www.pami.fudan.edu.cn/~jpzhang/)
and JianFeng Feng from Fudan Universiy.
[Junping Zhang](http://www.pami.fudan.edu.cn/~jpzhang/)
is the corresponding author.
The code is developed by
[Hanqing Chao](https://www.linkedin.com/in/hanqing-chao-9aa42412b/)
and [Yiwei He](https://www.linkedin.com/in/yiwei-he-4a6a6bbb/).
Currently, it is being maintained by
[Hanqing Chao](https://www.linkedin.com/in/hanqing-chao-9aa42412b/)
and Kun Wang.


## Citation
Please cite these papers in your publications if it helps your research:
```
@inproceedings{cao2017realtime,
  author = {Chao, Hanqing and He, Yiwei and Zhang, Junping and Feng, Jianfeng},
  booktitle = {AAAI},
  title = {{GaitSet}: Regarding Gait as a Set for Cross-View Gait Recognition},
  year = {2019}
}
```
Link to paper:
- [GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition](https://arxiv.org/abs/1811.06186)


## License
GaitSet is freely available for free non-commercial use, and may be redistributed under these conditions.
For commercial queries, contact [Junping Zhang](http://www.pami.fudan.edu.cn/~jpzhang/).