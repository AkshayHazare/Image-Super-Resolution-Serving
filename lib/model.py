from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.ops import *
import collections
import os
import math
import scipy.misc as sic
import numpy as np

# The inference data loader. Allow input image with different size
def inference_data_loader():
    if not os.path.isdir("./static/"):
        try: 
            os.mkdir("./static")
        except: None
    image_list_LR_temp = os.listdir("./static/")
    image_list_LR = [os.path.join("./static/", _) for _ in image_list_LR_temp if _.split('.')[-1] == 'png']

    # Read in and PreProcess the images
    def preprocess_test(name):
        im = sic.imread(name).astype(np.float32)

        # check GrayScale image
        if im.shape[-1] != 3:
            h, w = im.shape
            temp = np.empty((h, w, 3), dtype=np.uint8)
            temp[:, :, :] = im[:, :, np.newaxis]
            im = temp.copy()
        im = im / np.max(im)
        return im

    image_LR = [preprocess_test(_) for _ in image_list_LR]

    # Push path and image into a list
    Data = collections.namedtuple('Data', 'paths_LR, inputs')

    return Data(
        paths_LR=image_list_LR,
        inputs=image_LR
    )


# Definition of the generator
def generator(gen_inputs, gen_output_channels, reuse=False):

    # The Bx residual blocks
    def residual_block(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1')
            net = batchnorm(net, False)
            net = prelu_tf(net)
            net = conv2(net, 3, output_channel, stride, use_bias=False, scope='conv_2')
            net = batchnorm(net, False)
            net = net + inputs
        return net


    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv2(gen_inputs, 9, 64, 1, scope='conv')
            net = prelu_tf(net)

        stage1_output = net

        # The residual block parts
        for i in range(1, 16+1 , 1):
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = conv2(net, 3, 64, 1, use_bias=False, scope='conv')
            net = batchnorm(net, False)

        net = net + stage1_output

        with tf.variable_scope('subpixelconv_stage1'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('subpixelconv_stage2'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('output_stage'):
            net = conv2(net, 9, gen_output_channels, 1, scope='conv')
    return net

def save_images(fetches, step=None):
    image_dir = os.path.join("./output/", "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    filesets = []
    in_path = fetches['path_LR']
    name, _ = os.path.splitext(os.path.basename(str(in_path)))
    fileset = {"name": name, "step": step}
    kind = "outputs"
    filename = name + ".png"
    if step is not None:
        filename = "%08d-%s" % (step, filename)
    fileset[kind] = filename
    out_path = os.path.join(image_dir, filename)
    contents = fetches[kind][0]
    with open(out_path, "wb") as f:
        f.write(contents)
    filesets.append(fileset)
    return filesets
