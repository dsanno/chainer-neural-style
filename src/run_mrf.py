import argparse
import style_transfer_runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis')
    parser.add_argument('--model', '-m', default='vgg16.model',
                        help='model file path')
    parser.add_argument('--content', '-c', required=True,
                        help='Original image file path')
    parser.add_argument('--style', '-s', required=True,
                        help='Style image file path')
    parser.add_argument('--out_dir', '-o', default='output',
                        help='Output directory path')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--iter', default=100, type=int,
                        help='number of iteration')
    parser.add_argument('--save_iter', default=10, type=int,
                        help='number of iteration for saving images')
    parser.add_argument('--lr', default=2.0, type=float,
                        help='learning rate')
    parser.add_argument('--content_weight', default=1, type=float,
                        help='content image weight')
    parser.add_argument('--style_weight', default=0.2, type=float,
                        help='style image weight')
    parser.add_argument('--tv_weight', default=1e-5, type=float,
                        help='total variation weight')
    parser.add_argument('--width', '-w', default=256, type=int,
                        help='image width, height')
    parser.add_argument('--method', default='mrf', type=str, choices=['gram', 'mrf'],
                        help='style transfer method')
    args = parser.parse_args()

    style_transfer_runner.run(args)
