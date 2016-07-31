import argparse
import style_transfer_runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A Neural Algorithm of Artistic Style')
    parser.add_argument('--type', '-t', default='vgg16', choices=['vgg16', 'vgg19'],
                        help='model type')
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
    parser.add_argument('--iter', default=2000, type=int,
                        help='number of iteration for each resolution')
    parser.add_argument('--save_iter', default=100, type=int,
                        help='number of iteration for saving images')
    parser.add_argument('--lr', default=1.0, type=float,
                        help='learning rate')
    parser.add_argument('--content_weight', default=5, type=float,
                        help='content image weight')
    parser.add_argument('--style_weight', default=100, type=float,
                        help='style image weight')
    parser.add_argument('--tv_weight', default=1e-3, type=float,
                        help='total variation weight')
    parser.add_argument('--width', '-w', default=256, type=int,
                        help='image width, height')
    parser.add_argument('--method', default='gram', type=str, choices=['gram', 'mrf'],
                        help='style transfer method')
    parser.add_argument('--content_layers', default='3_3,4_3', type=str,
                        help='content layer names')
    parser.add_argument('--style_layers', default='1_2,2_2,3_3,4_3', type=str,
                        help='style layer names')
    parser.add_argument('--initial_image', default='random', type=str, choices=['content', 'random'],
                        help='initial image')
    parser.add_argument('--resolution_num', default=1, type=int, choices=[1,2,3],
                        help='the number of resolutions')
    parser.add_argument('--keep_color', action='store_true',
                        help='keep image color')
    parser.add_argument('--match_color_histogram', action='store_true',
                        help='use matching color histogram algorithm')
    parser.add_argument('--luminance_only', action='store_true',
                        help='use luminance only algorithm')
    args = parser.parse_args()

    style_transfer_runner.run(args)
