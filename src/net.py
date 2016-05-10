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

    def weights(self):
        return [0, 0, 0.5, 0.5], [0.25, 0.25, 0.25, 0.25]

    def __call__(self, x):
        y1 = F.relu(self.conv1_2(F.relu(self.conv1_1(x))))
        h = F.average_pooling_2d(y1, 2, stride=2)
        y2 = F.relu(self.conv2_2(F.relu(self.conv2_1(h))))
        h = F.average_pooling_2d(y2, 2, stride=2)
        y3 = F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(h))))))
        h = F.average_pooling_2d(y3, 2, stride=2)
        y4 = F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(h))))))
        return [y1, y2, y3, y4]
