from datetime import datetime
import numpy as np
import argparse

from model.initialization import initialization
from model.utils import evaluation
from config import conf

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='80000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
opt = parser.parse_args()


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result


m = initialization(conf, test=True)[0]

# load model checkpoint of iteration opt.iter
print('Loading the model of iteration %d...' % opt.iter)
m.load(opt.iter)
print('Transforming...')
time = datetime.now()
test = m.transform('test', opt.batch_size)
print('Evaluating...')
acc = evaluation(test, conf['data'])
print('Evaluation complete. Cost:', datetime.now() - time)

for i in range(1):
    print('===Rank-%d (Include identical-view cases)===' % (i + 1))
    print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
        np.mean(acc[0, :, :, i]),
        np.mean(acc[1, :, :, i]),
        np.mean(acc[2, :, :, i])))

# Print rank-1 accuracy of the best model
# e.g.
# =========================TOP-1=========================
# 87.23945454545456 70.35454545454546 94.96363636363635
for i in range(1):
    print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
    print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
        de_diag(acc[0, :, :, i]),
        de_diag(acc[1, :, :, i]),
        de_diag(acc[2, :, :, i])))

# Print rank-1 accuracy of the best model (Each Angle)
# e.g.
# =========================TOP-1=========================
# [83.8 91.2 91.8 88.789 83.3 81. 84.1 90. 92.2 94.445 79.]
# [61.4 75.4 80.7 77.3 72.1 70.1 71.5 73.5 73.5 68.4 50. ]
# [90.8 97.9 99.4 96.9 93.6 91.7 95.  97.8 98.9 96.8 85.8]
np.set_printoptions(precision=2, floatmode='fixed')
for i in range(1):
    print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
    print('NM:', de_diag(acc[0, :, :, i], True))
    print('BG:', de_diag(acc[1, :, :, i], True))
    print('CL:', de_diag(acc[2, :, :, i], True))
