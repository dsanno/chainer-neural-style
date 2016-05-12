import numpy as np
import six
import chainer
from chainer import cuda
from chainer import functions as F
from chainer import Variable
import cupy

import util

class NeuralStyle(object):
    def __init__(self, model, optimizer, content_weight, style_weight, tv_weight, device_id=-1):
        self.model = model
        self.optimizer = optimizer
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.device_id = device_id
        if device_id >= 0:
            self.xp = cuda.cupy
            self.model.to_gpu(device_id)
        else:
            self.xp = np

    def fit(self, content_image, style_image, epoch_num, callback=None):
        if self.device_id >= 0:
            with cupy.cuda.Device(self.device_id):
                return self.__fit(content_image, style_image, epoch_num, callback)
        else:
            return self.__fit(content_image, style_image, epoch_num, callback)

    def __fit(self, content_image, style_image, epoch_num, callback=None):
        xp = self.xp
        content_x = Variable(xp.asarray(content_image), volatile=True)
        style_x = Variable(xp.asarray(style_image), volatile=True)
        content_layer_names = ['3_3', '4_3']
        content_layers = self.model(content_x)
        content_layers = [(name, content_layers[name]) for name in content_layer_names]
        style_layer_names = ['1_2', '2_2', '3_3', '4_3']
        style_layers = self.model(style_x)
        style_grams = [(name, util.gram_matrix(style_layers[name])) for name in style_layer_names]
        link = chainer.Link(x=content_image.shape)
        if self.device_id >= 0:
            link.to_gpu()
        link.x.data = xp.random.uniform(-20, 20, size=content_image.shape).astype(np.float32)
        self.optimizer.setup(link)
        for epoch in six.moves.range(epoch_num):
            loss_info = self.__fit_one(link, content_layers, style_grams)
            if callback:
                callback(epoch, link.x, loss_info)
        return link.x

    def __fit_one(self, link, content_layers, style_grams):
        xp = self.xp
        link.zerograds()
        layers = self.model(link.x)
        loss_info = []
        loss = Variable(xp.zeros((), dtype=np.float32))
        for name, content_layer in content_layers:
            layer = layers[name]
            content_loss = self.content_weight * F.mean_squared_error(layer, Variable(content_layer.data))
            loss_info.append(('content_' + name, float(content_loss.data)))
            loss += content_loss
        for name, style_gram in style_grams:
            gram = util.gram_matrix(layers[name])
            style_loss = self.style_weight * F.mean_squared_error(gram, Variable(style_gram.data))
            loss_info.append(('style_' + name, float(style_loss.data)))
            loss += style_loss
        tv_loss = self.tv_weight * util.total_variation(link.x)
        loss_info.append(('tv', float(tv_loss.data)))
        loss += tv_loss
        loss.backward()
        self.optimizer.update()
        return loss_info

class MRF(object):
    def __init__(self, model, optimizer, content_weight, style_weight, tv_weight, device_id=-1):
        self.model = model
        self.optimizer = optimizer
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.device_id = device_id
        if device_id >= 0:
            self.xp = cuda.cupy
            self.model.to_gpu(device_id)
        else:
            self.xp = np

    def fit(self, content_image, style_image, epoch_num, callback=None):
        if self.device_id >= 0:
            with cupy.cuda.Device(self.device_id):
                return self.__fit(content_image, style_image, epoch_num, callback)
        else:
            return self.__fit(content_image, style_image, epoch_num, callback)

    def __fit(self, content_image, style_image, epoch_num, callback=None):
        xp = self.xp
        content_x = Variable(xp.asarray(content_image), volatile=True)
        style_x = Variable(xp.asarray(style_image), volatile=True)
        content_layer_names = ['4_2']
        content_layers = self.model(content_x)
        content_layers = [(name, content_layers[name]) for name in content_layer_names]
        style_layer_names = ['3_1', '4_1']
        style_layers = self.model(style_x)
        style_patches = []
        for name in style_layer_names:
            patch = util.patch(style_layers[name])
            patch_norm = F.expand_dims(F.sum(patch ** 2, axis=1) ** 0.5, 1)
            style_patches.append((name, patch, patch_norm))
        link = chainer.Link(x=content_image.shape)
        if self.device_id >= 0:
            link.to_gpu()
        link.x.data[:] = xp.asarray(content_image)
        self.optimizer.setup(link)
        for epoch in six.moves.range(epoch_num):
            loss_info = self.__fit_one(link, content_layers, style_patches)
            if callback:
                callback(epoch, link.x, loss_info)
        return link.x

    def __fit_one(self, link, content_layers, style_patches):
        xp = self.xp
        link.zerograds()
        layers = self.model(link.x)
        loss_info = []
        loss = Variable(xp.zeros((), dtype=np.float32))
        for name, content_layer in content_layers:
            layer = layers[name]
            content_loss = self.content_weight * F.mean_squared_error(layer, Variable(content_layer.data))
            loss_info.append(('content_' + name, float(content_loss.data)))
            loss += content_loss
        for name, style_patch, style_patch_norm in style_patches:
            patch = util.patch(layers[name])
            style_loss = self.style_weight * F.mean_squared_error(patch, util.nearest_neighbor_patch(patch, style_patch, style_patch_norm))
            loss_info.append(('style_' + name, float(style_loss.data)))
            loss += style_loss
        tv_loss = self.tv_weight * util.total_variation(link.x)
        loss_info.append(('tv', float(tv_loss.data)))
        loss += tv_loss
        loss.backward()
        self.optimizer.update()
        return loss_info
