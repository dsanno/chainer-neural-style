import numpy as np

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import Variable

def total_variation(x):
    xp = cuda.get_array_module(x.data)
    b, ch, h, w = x.data.shape
    wh = Variable(xp.asarray([[[[1], [-1]]]], dtype=np.float32).repeat(ch, 0).repeat(ch, 1), volatile='auto')
    ww = Variable(xp.asarray([[[[1, 1]]]], dtype=np.float32).repeat(ch, 0).repeat(ch, 1), volatile='auto')
    return F.sum(F.convolution_2d(x, W=wh) ** 2) / ((h - 1) * w) + F.sum(F.convolution_2d(x, W=ww) ** 2) / (h * (w - 1))

def gram_matrix(x):
    b, ch, h, w = x.data.shape
    v = F.reshape(x, (b, ch, w * h))
    return F.batch_matmul(v, v, transb=True) / np.float32(ch * w * h)
