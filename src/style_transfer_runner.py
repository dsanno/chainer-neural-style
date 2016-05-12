import argparse
import numpy as np
import os
from PIL import Image
import chainer
from chainer import functions as F
from chainer import cuda, optimizers, serializers
from util import total_variation, gram_matrix

from neural_style import NeuralStyle, MRF
from net import VGG

def open_and_resize_image(path, target_width, model):
    image = Image.open(path).convert('RGB')
    width, height = image.size
    target_height = int(round(float(height * target_width) / width))
    image = image.resize((target_width, target_height))
    return np.expand_dims(model.preprocess(np.asarray(image, dtype=np.float32), input_type='RGB'), 0)

def run(args):
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

    def on_epoch_done(epoch, x, losses):
        if (epoch + 1) % args.save_iter == 0:
            image = vgg.postprocess(cuda.to_cpu(x.data)[0], output_type='RGB').clip(0, 255).astype(np.uint8)
            Image.fromarray(image).save(os.path.join(args.out_dir, 'out_{0:04d}.png'.format(epoch + 1)))
            print 'epoch {} done'.format(epoch + 1)
            print 'losses:'
            label_width = max(map(lambda (name, loss): len(name), losses))
            for name, loss in losses:
                print '  {0:{width}s}: {1:f}'.format(name, loss, width=label_width)

    if args.method == 'mrf':
        model = MRF(vgg, optimizer, args.content_weight, args.style_weight, args.tv_weight, args.gpu)
    else:
        model = NeuralStyle(vgg, optimizer, args.content_weight, args.style_weight, args.tv_weight, args.gpu)
    out_image = model.fit(content_image, style_image, args.iter, on_epoch_done)
    image = vgg.postprocess(cuda.to_cpu(out_image.data)[0], output_type='RGB').clip(0, 255).astype(np.uint8)
    Image.fromarray(image).save(os.path.join(args.out_dir, 'out.png'))
