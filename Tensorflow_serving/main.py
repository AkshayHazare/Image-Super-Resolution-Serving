
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from lib.model import  generator, inference_data_loader, save_images
from lib.ops import *
import math
import time
import numpy as np


# Declare the test data reader
inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw')
path_LR = tf.placeholder(tf.string, shape=[], name='path_LR')

with tf.variable_scope('generator'):
        gen_output = generator(inputs_raw, 3, reuse=False)

with tf.name_scope('convert_image'):
    # DeProcess the images output from the model
    inputs = deprocessLR(inputs_raw)
    outputs = deprocess(gen_output)

    # Convert back to uint8
    converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
    converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

with tf.name_scope('encode_image'):
    save_fetch = {
        "path_LR": path_LR,
        "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
        "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs')
    }

# Define the weight initializer (In inference time, we only need to restore the weight of the generator)
var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
weight_initializer = tf.train.Saver(var_list)

# Define the initialization operation
init_op = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # Load the preTrained model
    # print('Loading weights from the pre-trained model')
    weight_initializer.restore(sess, "./Model/model-200000")
    x = []
    print('Evaluation starts!!')
    while(True):
        inference_data = inference_data_loader()
        for i in range(len(inference_data.inputs)):
                y = time.time()
                input_im = np.array([inference_data.inputs[i]]).astype(np.float32)
                path_lr = inference_data.paths_LR[i]
                results = sess.run(save_fetch, feed_dict={inputs_raw: input_im, path_LR: path_lr})
                filesets = save_images(results)
                print (time.time()-y)
                for i, f in enumerate(filesets):
                    print('evaluate image', f['name'])
                break
