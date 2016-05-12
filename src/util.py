import numpy as np

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import Variable

def total_variation(x):
    xp = cuda.get_array_module(x.data)
    b, ch, h, w = x.data.shape
    wh = Variable(xp.asarray([[[[1], [-1]]]], dtype=np.float32).repeat(ch, 0).repeat(ch, 1), volatile=x.volatile)
    ww = Variable(xp.asarray([[[[1, 1]]]], dtype=np.float32).repeat(ch, 0).repeat(ch, 1), volatile=x.volatile)
    return F.sum(F.convolution_2d(x, W=wh) ** 2) / ((h - 1) * w) + F.sum(F.convolution_2d(x, W=ww) ** 2) / (h * (w - 1))

def gram_matrix(x):
    b, ch, h, w = x.data.shape
    v = F.reshape(x, (b, ch, w * h))
    return F.batch_matmul(v, v, transb=True) / np.float32(ch * w * h)

def patch(x, ksize=3, stride=1, pad=0):
    xp = cuda.get_array_module(x.data)
    b, ch, h, w = x.data.shape
    w = Variable(xp.identity(ch * ksize * ksize, dtype=np.float32).reshape((ch * ksize * ksize, ch, ksize, ksize)), volatile=x.volatile)
    return F.convolution_2d(x, W=w, stride=stride, pad=pad)

def nearest_neighbor_patch(x, patch, patch_norm):
    assert patch.data.shape[0] == 1, 'mini batch size of patch must be 1'
    assert patch_norm.data.shape[0] == 1, 'mini batch size of patch_norm must be 1'

    xp = cuda.get_array_module(x.data)
    z = x.data
    b, ch, h, w = z.shape
    z = z.transpose((1, 0, 2, 3)).reshape((ch, -1))
    norm = xp.expand_dims(xp.sum(z ** 2, axis=0) ** 0.5, 0)
    z = z / xp.broadcast_to(norm, z.shape)
    p = patch.data
    p_norm = patch_norm.data
    p = p.reshape((ch, -1))
    p_norm = p_norm.reshape((1, -1))
    p_normalized = p / xp.broadcast_to(p_norm, p.shape)
    correlation = z.T.dot(p_normalized)
    min_index = xp.argmax(correlation, axis=1)
    nearest_neighbor = p.take(min_index, axis=1).reshape((ch, b, h, w)).transpose((1, 0, 2, 3))
    return Variable(nearest_neighbor, volatile=x.volatile)
