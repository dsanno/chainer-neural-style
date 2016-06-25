import argparse
import numpy as np
from scipy import linalg
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer image using Image Analogy color transfer')
    parser.add_argument('style_image', type=str, help='Style image file path')
    parser.add_argument('content_image', type=str, help='Content image file path')
    parser.add_argument('output_image', type=str, help='Output image file path')
    args = parser.parse_args()

    xs = np.asarray(Image.open(args.style_image).convert('RGB'), dtype=np.float32)
    xc = np.asarray(Image.open(args.content_image).convert('RGB'), dtype=np.float32)
    style_shape = xs.shape

    xs = xs.reshape((-1, 3)).transpose((1, 0))
    xs_mean = np.mean(xs, axis=1, keepdims=True)
    xs_var = np.cov(xs)
    d, v = linalg.eig(xs_var)
    sigma_style_inv = v.dot(np.diag(d ** (-0.5))).dot(v.T)

    xc = xc.reshape((-1, 3)).transpose((1, 0))
    xc_mean = np.mean(xc, axis=1, keepdims=True)
    xc_var = np.cov(xc)
    d, v = linalg.eig(xc_var)
    sigma_content = v.dot(np.diag(d ** 0.5)).dot(v.T)

    a = sigma_content.dot(sigma_style_inv)
    b = xc_mean - a.dot(xs_mean)
    xs2 = a.dot(xs) + b
    xs2 = xs2.transpose((1, 0)).reshape(style_shape).real.clip(0, 255).astype(np.uint8)

    Image.fromarray(xs2).save(args.output_image)
