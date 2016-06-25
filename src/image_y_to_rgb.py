import argparse
import colorsys
import numpy as np
import six
from scipy import linalg
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer image using Image Analogy color transfer')
    parser.add_argument('luminance_image', type=str, help='Luminance image file path')
    parser.add_argument('color_image', type=str, help='Color image file path')
    parser.add_argument('output_image', type=str, help='Output image file path')
    args = parser.parse_args()

    rgb_to_l = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    xl = np.asarray(Image.open(args.luminance_image).convert('RGB'), dtype=np.float32) / 255
    content_shape = xl.shape
    xc = np.asarray(Image.open(args.color_image).convert('RGB').resize(content_shape[:2]), dtype=np.float32) / 255

    xl = xl.reshape((-1, 3))
    for i in six.moves.range(len(xl)):
        xl[i,:] = colorsys.rgb_to_yiq(*xl[i])

    xc = xc.reshape((-1, 3))
    for i in six.moves.range(len(xc)):
        xc[i,:] = colorsys.rgb_to_yiq(*xc[i])
    xc[:, 0] = xl[:, 0]

    for i in six.moves.range(len(xc)):
        xc[i,:] = colorsys.yiq_to_rgb(*xc[i])

    xc = xc * 255
    xc = xc.clip(0, 255).astype(np.uint8).reshape(content_shape)
    Image.fromarray(xc).save(args.output_image)
