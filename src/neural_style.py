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
                self.__fit(content_image, style_image, epoch_num, callback)
        else:
            self.__fit(content_image, style_image, epoch_num, callback)

    def __fit(self, content_image, style_image, epoch_num, callback=None):
        xp = self.xp
        content_x = Variable(xp.asarray(content_image), volatile=True)
        style_x = Variable(xp.asarray(style_image), volatile=True)
        content_layers = [layer for layer in self.model(content_x)]
        style_grams = [util.gram_matrix(layer) for layer in self.model(style_x)]
        link = chainer.Link(x=content_image.shape)
        if self.device_id >= 0:
            link.to_gpu()
        link.x.data = xp.random.uniform(-20, 20, size=content_image.shape).astype(np.float32)
        self.optimizer.setup(link)
        for epoch in six.moves.range(epoch_num):
            content_losses, style_losses = self.__fit_one(link, (content_layers, style_grams))
            if callback:
                callback(epoch, link.x, content_losses, style_losses)

    def __fit_one(self, link, target):
        xp = self.xp
        link.zerograds()
        content_layers, style_grams = target
        content_coeffs, style_coeffs = self.model.weights()
        layers = self.model(link.x)
        content_losses = []
        style_losses = []
        loss = Variable(xp.zeros((), dtype=np.float32))
        params = zip(layers, content_layers, style_grams, content_coeffs, style_coeffs)
        for layer, content_layer, style_gram, content_coeff, style_coeff in params:
            if content_coeff > 0:
                content_loss = self.content_weight * content_coeff * F.mean_squared_error(layer, Variable(content_layer.data))
            else:
                content_loss = Variable(xp.zeros((), dtype=np.float32))
            gram = util.gram_matrix(layer)
            if style_coeff > 0:
                style_loss = self.style_weight * style_coeff * F.mean_squared_error(gram, Variable(style_gram.data))
            else:
                style_loss = Variable(xp.zeros((), dtype=np.float32))
            loss += content_loss + style_loss
            content_losses.append(float(content_loss.data))
            style_losses.append(float(style_loss.data))
        loss += self.tv_weight * util.total_variation(link.x)
        loss.backward()
        self.optimizer.update()
        return (content_losses, style_losses)
