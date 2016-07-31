import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L

class VGG(chainer.Chain):
    def __init__(self):
        super(VGG, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
        )
        self.mean = np.asarray([104, 117, 124], dtype=np.float32)

    def preprocess(self, image, input_type='BGR'):
        if input_type == 'RGB':
            image = image[:,:,::-1]
        return np.rollaxis(image - self.mean, 2)

    def postprocess(self, image, output_type='RGB'):
        image = np.transpose(image, (1, 2, 0)) + self.mean
        if output_type == 'RGB':
            return image[:,:,::-1]
        else:
            return image

    def __call__(self, x):
        layer_names = ['1_1', '1_2', 'pool', '2_1', '2_2', 'pool', '3_1', '3_2', '3_3', 'pool', '4_1', '4_2', '4_3']
        layers = {}
        h = x
        for layer_name in layer_names:
            if layer_name == 'pool':
                h = F.max_pooling_2d(h, 2, stride=2)
            else:
                h = F.relu(self['conv' + layer_name](h))
                layers[layer_name] = h
        return layers

class VGG19(chainer.Chain):
    def __init__(self):
        super(VGG19, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_4=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_4=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_4=L.Convolution2D(512, 512, 3, stride=1, pad=1),
        )
        self.mean = np.asarray([104, 117, 124], dtype=np.float32)

    def preprocess(self, image, input_type='BGR'):
        if input_type == 'RGB':
            image = image[:,:,::-1]
        return np.rollaxis(image - self.mean, 2)

    def postprocess(self, image, output_type='RGB'):
        image = np.transpose(image, (1, 2, 0)) + self.mean
        if output_type == 'RGB':
            return image[:,:,::-1]
        else:
            return image

    def __call__(self, x):
        layer_names = ['1_1', '1_2', 'pool', '2_1', '2_2', 'pool', '3_1',
                       '3_2', '3_3', '3_4', 'pool', '4_1', '4_2', '4_3', '4_4',
                       'pool', '5_1', '5_2', '5_3', '5_4']
        layers = {}
        h = x
        for layer_name in layer_names:
            if layer_name == 'pool':
                h = F.max_pooling_2d(h, 2, stride=2)
            else:
                h = F.relu(self['conv' + layer_name](h))
                layers[layer_name] = h
        return layers
