import argparse
import colorsys
import numpy as np
import six
from scipy import linalg
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer image using Image Analogy color transfer')
    parser.add_argument('style_image', type=str, help='Style image file path')
    parser.add_argument('content_image', type=str, help='Content image file path')
    parser.add_argument('output_style_image', type=str, help='Output image file path')
    parser.add_argument('output_content_image', type=str, help='Output image file path')
    args = parser.parse_args()

    xs = np.asarray(Image.open(args.style_image).convert('RGB'), dtype=np.float32) / 255
    xc = np.asarray(Image.open(args.content_image).convert('RGB'), dtype=np.float32) / 255
    style_shape = xs.shape
    content_shape = xc.shape

    xs = xs.reshape((-1, 3))
    for i in six.moves.range(len(xs)):
        xs[i,:] = colorsys.rgb_to_yiq(*xs[i])
    ys = xs[:, 0:1]
    ys_mean = np.mean(ys)
    ys_std = np.std(ys)

    xc = xc.reshape((-1, 3))
    for i in six.moves.range(len(xc)):
        xc[i,:] = colorsys.rgb_to_yiq(*xc[i])
    yc = xc[:, 0:1]
    yc_mean = np.mean(yc)
    yc_std = np.std(yc)

    ys = (ys - ys_mean) / ys_std * yc_std + yc_mean

    xs2 = (ys * 255).repeat(3, axis=1).clip(0, 255).astype(np.uint8).reshape(style_shape)
    Image.fromarray(xs2).save(args.output_style_image)

    xc2 = (yc * 255).repeat(3, axis=1).clip(0, 255).astype(np.uint8).reshape(content_shape)
    Image.fromarray(xc2).save(args.output_content_image)
