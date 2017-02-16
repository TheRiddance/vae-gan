""" GAN """

import os
import tensorflow as tf

import model_ops
from visualizer import ImageVisualizer


class GAN:

    def __init__(self, batch_size, hidden_size=256, learning_rate=1e-3, beta1=0.5, image_size=64):
        self.image_size = image_size
        self.input_tensor = tf.placeholder(tf.float32, [None, image_size * image_size])

        with tf.variable_scope("model"):
            self.G = self._generator(tf.random_normal([batch_size, hidden_size]), batch_size)
            self.D1_logits = self._discriminator(self.input_tensor, batch_size)

        with tf.variable_scope("model", reuse=True):
            self.D2_logits = self._discriminator(self.G, batch_size)
            # sample uniformly from [-1, 1] or normal [0, 1] ?
            self.sampler = self._generator(tf.random_normal([batch_size, hidden_size]), batch_size)

        with tf.variable_scope("loss"):
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D1_logits, tf.ones_like(self.D1_logits)))
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D2_logits, tf.zeros_like(self.D2_logits)))
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D2_logits, tf.ones_like(self.D2_logits)))

            self.d_loss = self.d_loss_real + self.d_loss_fake

            train_variables = tf.trainable_variables()
            self.d_vars = [var for var in train_variables if "dis_" in var.name]
            self.g_vars = [var for var in train_variables if "gen_" in var.name]

            self.d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.d_loss, var_list=self.d_vars)
            self.g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.g_loss, var_list=self.g_vars)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _discriminator(self, input_tensor, batch_size, output_size=1, image_size=64):

        net = tf.reshape(input_tensor, [batch_size, image_size, image_size, 1])
        net = model_ops.conv2d(net, 32, stride=2, name="dis_conv1")
        net = model_ops.conv2d(net, 64, stride=2, name="dis_conv2")
        net = model_ops.conv2d(net, 128, stride=2, padding='VALID', name="dis_conv3")
        net = tf.nn.dropout(net, keep_prob=0.9)
        net = tf.reshape(net, [batch_size, -1])
        return model_ops.linear(net, output_size, "dis_fully")

    def _generator(self, input_tensor, batch_size):

        net = tf.expand_dims(input_tensor, 1)
        net = tf.expand_dims(net, 1)
        net = model_ops.conv2d_back(net, [batch_size, 8, 8, 128], kernel=8, padding='VALID', name="gen_conv1")
        net = model_ops.conv2d_back(net, [batch_size, 16, 16, 64], stride=2, name="gen_conv2")
        net = model_ops.conv2d_back(net, [batch_size, 32, 32, 32], stride=2, name="gen_conv3")
        net = model_ops.conv2d_back(net, [batch_size, 64, 64, 1], stride=2, name="gen_conv4")
        net = tf.nn.tanh(net)
        return tf.reshape(net, [batch_size, -1])

    def update_params(self, input_tensor):
        # Update D network
        _ = self.sess.run([self.d_optim], feed_dict={self.input_tensor: input_tensor})

        # Update G network
        _ = self.sess.run([self.g_optim])

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _ = self.sess.run([self.g_optim])

        d_loss, g_loss = self.sess.run([self.d_loss, self.g_loss], feed_dict={self.input_tensor: input_tensor})
        return d_loss, g_loss

    def generate_and_save_images(self, num_samples, directory, epoch):
        # create experiment folder
        experiment_dir = os.path.join(directory, "GAN")
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
            print('created directory:', experiment_dir)

        generated_samples = self.sess.run(self.sampler)
        visualizer = ImageVisualizer(experiment_dir, image_size=self.image_size)
        visualizer.save_generated_samples(generated_samples, epoch)

    def print_loss(self, input_tensor, epoch):
        d_loss, g_loss = self.sess.run([self.d_loss, self.g_loss], feed_dict={self.input_tensor: input_tensor})
        print("epoch: %3d" % epoch, "D loss %.4f" % d_loss, "G loss  %.4f" % g_loss)
