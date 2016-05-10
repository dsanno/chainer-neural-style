import argparse
import numpy as np
import os
from PIL import Image
import chainer
from chainer import functions as F
from chainer import cuda, optimizers, serializers
from util import total_variation, gram_matrix

from neural_style import NeuralStyle
from net import VGG

def open_and_resize_image(path, target_width, model):
    image = Image.open(path).convert('RGB')
    width, height = image.size
    target_height = int(round(float(height * target_width) / width))
    image = image.resize((target_width, target_height))
    return np.expand_dims(model.preprocess(np.asarray(image, dtype=np.float32), input_type='RGB'), 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A Neural Algorithm of Artistic Style')
    parser.add_argument('--model', '-m', default='vgg.model',
                        help='model file path')
    parser.add_argument('--content', '-c', required=True,
                        help='Original image file path')
    parser.add_argument('--style', '-s', required=True,
                        help='Style image file path')
    parser.add_argument('--out_dir', '-o', default='output',
                        help='Output directory path')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--iter', default=2000, type=int,
                        help='number of iteration')
    parser.add_argument('--lr', default=10.0, type=float,
                        help='learning rate')
    parser.add_argument('--content_weight', default=0.005, type=float,
                        help='content image weight')
    parser.add_argument('--style_weight', default=1, type=float,
                        help='style image weight')
    parser.add_argument('--tv_weight', default=1e-5, type=float,
                        help='total variation weight')
    parser.add_argument('--width', '-w', default=256, type=int,
                        help='image width, height')
    args = parser.parse_args()

    if args.out_dir != None:
        if not os.path.exists(args.out_dir):
            try:
                os.mkdir(args.out_dir)
            except:
                print 'cannot make directory {}'.format(args.out_dir)
                exit()
        elif not os.path.isdir(args.out_dir):
            print 'file path {} exists but is not directory'.format(args.out_dir)
            exit()
    vgg = VGG()
    content_image = open_and_resize_image(args.content, args.width, vgg)
    print 'loading content image completed'
    style_image = open_and_resize_image(args.style, args.width, vgg)
    print 'loading style image completed'
    serializers.load_hdf5(args.model, vgg)
    print 'loading neural network model completed'
    optimizer = optimizers.Adam(alpha=args.lr)

    def on_epoch_done(epoch, x, content_losses, style_losses):
        if (epoch + 1) % 100 == 0:
            image = vgg.postprocess(cuda.to_cpu(x.data)[0], output_type='RGB').clip(0, 255).astype(np.uint8)
            Image.fromarray(image).save(os.path.join(args.out_dir, 'out_{0:04d}.jpg'.format(epoch + 1)))
            print 'epoch {} done'.format(epoch + 1)
            for i, (content_loss, style_loss) in enumerate(zip(content_losses, style_losses)):
                print 'layer: {} content_loss: {} style_loss: {}'.format(i, content_loss, style_loss)

    model = NeuralStyle(vgg, optimizer, args.content_weight, args.style_weight, args.tv_weight, args.gpu)
    model.fit(content_image, style_image, args.iter, on_epoch_done)
