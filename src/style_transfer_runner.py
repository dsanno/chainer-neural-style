import argparse
import numpy as np
import os
from PIL import Image
import chainer
from chainer import functions as F
from chainer import cuda, optimizers, serializers
import util

from neural_style import NeuralStyle, MRF
from net import VGG, VGG19
from lbfgs import LBFGS

def open_and_resize_image(path, target_width, model):
    image = Image.open(path).convert('RGB')
    width, height = image.size
    target_height = int(round(float(height * target_width) / width))
    image = image.resize((target_width, target_height), Image.BILINEAR)
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
    if args.type == 'vgg19':
        vgg = VGG19()
    else:
        vgg = VGG()
    content_image = open_and_resize_image(args.content, args.width, vgg)
    print 'loading content image completed'
    style_image = open_and_resize_image(args.style, args.width, vgg)
    if args.match_color_histogram:
        style_image = util.match_color_histogram(style_image, content_image)
    if args.luminance_only:
        content_image, content_iq = util.split_bgr_to_yiq(content_image)
        style_image, style_iq = util.split_bgr_to_yiq(style_image)
        content_mean = np.mean(content_image, axis=(1,2,3), keepdims=True)
        content_std = np.std(content_image, axis=(1,2,3), keepdims=True)
        style_mean = np.mean(style_image, axis=(1,2,3), keepdims=True)
        style_std = np.std(style_image, axis=(1,2,3), keepdims=True)
        style_image = (style_image - style_mean) / style_std * content_std + content_mean
    print 'loading style image completed'
    serializers.load_hdf5(args.model, vgg)
    print 'loading neural network model completed'
    optimizer = LBFGS(args.lr, size=10)
    content_layers = args.content_layers.split(',')
    style_layers = args.style_layers.split(',')

    def on_epoch_done(epoch, x, losses):
        if (epoch + 1) % args.save_iter == 0:
            image = cuda.to_cpu(x.data)
            if args.luminance_only:
                image = util.join_yiq_to_bgr(image, content_iq)
            image = vgg.postprocess(image[0], output_type='RGB').clip(0, 255).astype(np.uint8)
            Image.fromarray(image).save(os.path.join(args.out_dir, 'out_{0:04d}.png'.format(epoch + 1)))
            print 'epoch {} done'.format(epoch + 1)
            print 'losses:'
            label_width = max(map(lambda (name, loss): len(name), losses))
            for name, loss in losses:
                print '  {0:{width}s}: {1:f}'.format(name, loss, width=label_width)

    if args.method == 'mrf':
        model = MRF(vgg, optimizer, args.content_weight, args.style_weight, args.tv_weight, content_layers, style_layers, args.resolution_num, args.gpu, initial_image=args.initial_image, keep_color=args.keep_color)
    else:
        model = NeuralStyle(vgg, optimizer, args.content_weight, args.style_weight, args.tv_weight, content_layers, style_layers, args.resolution_num, args.gpu, initial_image=args.initial_image, keep_color=args.keep_color)
    out_image = model.fit(content_image, style_image, args.iter, on_epoch_done)
    out_image = cuda.to_cpu(out_image.data)
    if args.luminance_only:
        out_image = util.join_yiq_to_bgr(out_image, content_iq)
    image = vgg.postprocess(out_image[0], output_type='RGB').clip(0, 255).astype(np.uint8)
    Image.fromarray(image).save(os.path.join(args.out_dir, 'out.png'))
