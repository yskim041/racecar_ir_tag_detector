# tf_utils.py

import tensorflow as tf


def conv_down(x, ch_in, ch_out=None, ksize=3):
    if ch_out is None:
        ch_out = ch_in * 2

    return max_pool_2x2(conv(x, ch_in, ch_out, strides=[1, 1, 1, 1],
                             ksize=ksize))


def conv(x, ch_in, ch_out, strides=[1, 1, 1, 1], ksize=3, relu=True):
    W = weight_variable([ksize, ksize, ch_in, ch_out])
    b = bias_variable([ch_out])
    h_conv = conv2d(x, W, strides=strides) + b
    if relu:
        h_conv = tf.nn.relu(h_conv)
    return h_conv


def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv_flat(x, ch_in, ch_out):
    W = weight_variable([ch_in, ch_out])
    b = bias_variable([ch_out])
    h_reshape = tf.reshape(x, [-1, ch_in])
    h_flat = tf.nn.relu(tf.matmul(h_reshape, W) + b)
    return h_flat


def deconv_flat(x, ch_in, ch_out, batch_size):
    W = weight_variable([ch_in, ch_out[0] * ch_out[1] * ch_out[2]])
    b = bias_variable([ch_out[0] * ch_out[1] * ch_out[2]])
    h = tf.nn.relu(tf.matmul(x, W) + b)
    return tf.reshape(h, [batch_size] + ch_out)


def deconv_with_concat(x, xu, ch_in, out_shape, ch_mid=None, ksize=3):
    if ch_mid is None:
        ch_mid = ch_in
    dl = deconv(x, ch_in, out_shape,
                strides=[1, 2, 2, 1])
    dl = tf.concat([dl, xu], 3)
    return conv(dl, ch_mid, out_shape[3], strides=[1, 1, 1, 1], ksize=ksize)


def deconv(x, ch_in, out_shape, strides=[1, 2, 2, 1], ksize=3, relu=True):
    W = weight_variable([ksize, ksize, out_shape[3], ch_in])
    b = bias_variable([out_shape[3]])
    h_dconv = deconv2d(x, W, out_shape=out_shape, strides=strides) + b
    if relu:
        h_dconv = tf.nn.relu(h_dconv)
    return h_dconv


def deconv2d(x, W, out_shape, strides=[1, 2, 2, 1]):
    return tf.nn.conv2d_transpose(x, W, output_shape=out_shape,
                                  strides=strides, padding='SAME')


def fc_layer(x, ch_in, ch_out):
    W = weight_variable([ch_in, ch_out])
    b = bias_variable([ch_out])
    h_vec = tf.matmul(x, W) + b
    return h_vec


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# End of script
