# Chainer implementation of style transfer using neural network

Implementation of
* [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
* [Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis](http://arxiv.org/abs/1601.04589)

# Requirements

* Python 2.7
* [Chainer 1.8.1](http://chainer.org/)
* [Pillow](https://pypi.python.org/pypi/Pillow/)

# Usage

## Download VGG 16 layers caffe model

* Visit https://gist.github.com/ksimonyan/211839e770f7b538e2d8 and download VGG_ILSVRC_16_layers.caffemodel.
* Put downloaded file into this directory.

## Convert caffemodel to chainer model

```
$ python src/create_chainer_model.py
```

## Transfer image style using "A Neural Algorithm of Artistic Style"

```
$ python src/run.py -c content_image.png -s style_image.png -o out_dir -g 0
```

Options:

* -c (--content) \<file path\>: required  
Content image file path
* -s (--style) \<file path\>: required  
Style image file path
* -o (--out_dir) \<directory path\>: optional  
Output directory path (default: output)
* -g (--gpu) \<GPU device index\>: optional  
GPU device index. Negative value indecates CPU (default: -1)
* --w (--width) \<integer\>: optional  
Image width (default: 256)
* --iter \<integer\>: optional  
Number of iteration for each iteration (default: 2000)
* --initial_image \<string\|: optional
Initial image of optimization: "random" or "content" (default: random)
* --keep_color: optional
Keep color phase if specified
* --resolution_num \<int\>: optional  
Number of resolutions (default: 1)
* --save_iter \<integer\>: optional  
Number of iteration for saving images (default: 100)
* --lr \<float\>: optional  
Learning rate: "alpha" value of ADAM (default: 10)
* --content_weight \<float\>: optional  
Weight of content loss (default: 0.005)
* --style_weight \<float\>: optional  
Weight of style loss (default: 1)
* --tv_weight \<float\>: optional  
Weight of total variation loss (default: 1e-5)

## Transfer image style using Markov Random Fields algorithm

```
$ python src/run_mrf.py -c content_image.png -s style_image.png -o out_dir -g 0
```

Options:

* -c (--content) \<file path\>: required  
Content image file path
* -s (--style) \<file path\>: required  
Style image file path
* -o (--out_dir) \<directory path\>: optional  
Output directory path (default: output)
* -g (--gpu) \<GPU device index\>: optional  
GPU device index. Negative value indecates CPU (default: -1)
* --w (--width) \<integer\>: optional  
Image width (default: 256)
* --iter \<integer\>: optional  
Number of iteration for each resolution (default: 100)
* --initial_image \<string\|: optional
Initial image of optimization: "random" or "content" (default: content)
* --keep_color: optional
Keep color phase if specified
* --resolution_num \<int\>: optional  
Number of resolutions (default: 3)
* --save_iter \<integer\>: optional  
Number of iteration for saving images (default: 10)
* --lr \<float\>: optional  
Learning rate: "alpha" value of ADAM (default: 2.0)
* --content_weight \<float\>: optional  
Weight of content loss (default: 0.2)
* --style_weight \<float\>: optional  
Weight of style loss (default: 1)
* --tv_weight \<float\>: optional  
Weight of total variation loss (default: 1e-5)

# License

MIT License
