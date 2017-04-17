from tensorflow.contrib import layers
import tensorflow as tf


def batch_norm(x, epsilon=1e-5, momentum=0.9, name="batch_norm"):
    return layers.batch_norm(x,
                             decay=momentum,
                             updates_collections=None,
                             epsilon=epsilon,
                             scale=True,
                             scope=name)


def leaky_relu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def leaky_relu_batch_norm(x, alpha=0.2):
    return leaky_relu(batch_norm(x), alpha)


def conv2d(input_, output_dim, kernel=5, stride=1, stddev=0.02, padding='SAME', name="conv2d"):

    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [kernel, kernel, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, weights, strides=[1, stride, stride, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return tf.nn.elu(batch_norm(conv))


def conv2d_back(input_, output_shape, kernel=5, stride=1, stddev=0.02, padding='SAME', name="deconv2d"):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [kernel, kernel, output_shape[-1], input_.get_shape().as_list()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, weights, output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return tf.nn.sigmoid(batch_norm(deconv))


def linear(input_, output_size, scope="linear", stddev=0.02):
    with tf.variable_scope(scope):
        weights = tf.get_variable('weights', [input_.get_shape().as_list()[-1], output_size], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(0.0))

        return tf.matmul(input_, weights) + bias

def linear_contrib(inputs, output_size, activation_fn=None, scope="linear", stddev=0.02):
    with tf.variable_scope(scope):
        return layers.fully_connected(inputs, output_size,
                                      activation_fn=activation_fn,
                                      weights_initializer=tf.random_normal_initializer(stddev=stddev),
                                      biases_initializer=tf.constant_initializer(0.0))
