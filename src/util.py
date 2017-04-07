import numpy as np
import six

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import Variable

def total_variation(x):
    xp = cuda.get_array_module(x.data)
    b, ch, h, w = x.data.shape
    wh = Variable(xp.asarray([[[[1], [-1]], [[0], [0]], [[0], [0]]], [[[0], [0]], [[1], [-1]], [[0], [0]]], [[[0], [0]], [[0], [0]], [[1], [-1]]]], dtype=np.float32), volatile=x.volatile)
    ww = Variable(xp.asarray([[[[1, -1]], [[0, 0]], [[0, 0]]], [[[0, 0]], [[1, -1]], [[0, 0]]], [[[0, 0]], [[0, 0]], [[1, -1]]]], dtype=np.float32), volatile=x.volatile)
    return F.sum(F.convolution_2d(x, W=wh) ** 2) + F.sum(F.convolution_2d(x, W=ww) ** 2)

def gram_matrix(x):
    b, ch, h, w = x.data.shape
    v = F.reshape(x, (b, ch, w * h))
    return F.batch_matmul(v, v, transb=True) / np.float32(ch * w * h)

def patch(x, ksize=3, stride=1, pad=0):
    xp = cuda.get_array_module(x.data)
    b, ch, h, w = x.data.shape
    w = Variable(xp.identity(ch * ksize * ksize, dtype=np.float32).reshape((ch * ksize * ksize, ch, ksize, ksize)), volatile=x.volatile)
    return F.convolution_2d(x, W=w, stride=stride, pad=pad)

def gray(x):
    xp = cuda.get_array_module(x.data)
    w = Variable(xp.asarray([[[[0.114]], [[0.587]], [[0.299]]], [[[0.114]], [[0.587]], [[0.299]]], [[[0.114]], [[0.587]], [[0.299]]]], dtype=np.float32), volatile=x.volatile)
    return F.convolution_2d(x, W=w)

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

def luminance_only(x, y):
    xp = cuda.get_array_module(x)
    w = xp.asarray([0.114, 0.587, 0.299], dtype=np.float32)
    x_shape = x.shape
    y_shape = y.shape

    x = x.reshape(x_shape[:2] + (-1,))
    xl = xp.zeros((x.shape[0], 1, x.shape[2]), dtype=np.float32)
    for i in six.moves.range(len(x)):
        xl[i,:] = w.dot(x[i])
    xl_mean = xp.mean(xl, axis=2, keepdims=True)
    xl_std = xp.std(xl, axis=2, keepdims=True)

    y = y.reshape(y_shape[:2] + (-1,))
    yl = xp.zeros((y.shape[0], 1, y.shape[2]), dtype=np.float32)
    for i in six.moves.range(len(y)):
        yl[i,:] = w.dot(y[i])
    yl_mean = xp.mean(yl, axis=2, keepdims=True)
    yl_std = xp.std(yl, axis=2, keepdims=True)

    xl = (xl - xl_mean) / xl_std * yl_std + yl_mean
    return xp.repeat(xl, 3, axis=1).reshape(x_shape)

def bgr_to_yiq(x):
    transform = np.asarray([[0.114, 0.587, 0.299], [-0.322, -0.274, 0.596], [0.312, -0.523, 0.211]], dtype=np.float32)
    n, c, h, w = x.shape
    x = x.transpose((1, 0, 2, 3)).reshape((c, -1))
    x = transform.dot(x)
    return x.reshape((c, n, h, w)).transpose((1, 0, 2, 3))

def yiq_to_bgr(x):
    transform = np.asarray([[1, -1.106, 1.703], [1, -0.272, -0.647], [1, 0.956, 0.621]], dtype=np.float32)
    n, c, h, w = x.shape
    x = x.transpose((1, 0, 2, 3)).reshape((c, -1))
    x = transform.dot(x)
    return x.reshape((c, n, h, w)).transpose((1, 0, 2, 3))

def split_bgr_to_yiq(x):
    x = bgr_to_yiq(x)
    y = x[:,0:1,:,:]
    iq = x[:,1:,:,:]
    return np.repeat(y, 3, axis=1), iq

def join_yiq_to_bgr(y, iq):
    y = bgr_to_yiq(y)[:,0:1,:,:]
    return yiq_to_bgr(np.concatenate((y, iq), axis=1))

def match_color_histogram(x, y):
    z = np.zeros_like(x)
    shape = x[0].shape
    for i in six.moves.range(len(x)):
        a = x[i].reshape((3, -1))
        a_mean = np.mean(a, axis=1, keepdims=True)
        a_var = np.cov(a)
        d, v = np.linalg.eig(a_var)
        d += 1e-6
        a_sigma_inv = v.dot(np.diag(d ** (-0.5))).dot(v.T)

        b = y[i].reshape((3, -1))
        b_mean = np.mean(b, axis=1, keepdims=True)
        b_var = np.cov(b)
        d, v = np.linalg.eig(b_var)
        b_sigma = v.dot(np.diag(d ** 0.5)).dot(v.T)

        transform = b_sigma.dot(a_sigma_inv)
        z[i,:] = (transform.dot(a - a_mean) + b_mean).reshape(shape)
    return z
