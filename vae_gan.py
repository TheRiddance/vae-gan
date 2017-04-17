"""
    VAE-GAN: combination of Encoder, Generator and Discriminator network

    based on: https://arxiv.org/pdf/1512.09300.pdf
"""

import os
from tensorflow.contrib import layers
import tensorflow as tf
import numpy as np

import model_ops


class VAE_GAN:

    def __init__(self, hidden_size, image_size=64, channels=3, experiment_name="VAE_GAN"):
        self.experiment_dir = experiment_name

        self.d_real = 0
        self.d_fake = 0

        self.e_learning_rate = 1e-3
        self.g_learning_rate = 1e-3
        self.d_learning_rate = 1e-3

        self.hidden_size = hidden_size
        self.image_size = image_size
        self.channels = channels

        self.lr_D = tf.placeholder(tf.float32, shape=[])
        self.lr_G = tf.placeholder(tf.float32, shape=[])
        self.lr_E = tf.placeholder(tf.float32, shape=[])

        self.x = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.channels])
        self.batch_size = tf.shape(self.x)[0]

        self.z_p = tf.random_normal((self.batch_size, self.hidden_size), 0, 1)  # normal dist for GAN
        self.eps = tf.random_normal((self.batch_size, self.hidden_size), 0, 1)  # normal dist for VAE

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope("enc"):
            self.z_x_mean, self.z_x_log_sigma_sq = self._encoder(self.x, self.hidden_size)  # get z from the input

        with tf.variable_scope("gen"):
            self.z_x = tf.add(self.z_x_mean, tf.multiply(tf.sqrt(tf.exp(self.z_x_log_sigma_sq)), self.eps))  # grab our actual z
            self.x_tilde = self._generator(self.z_x)

        with tf.variable_scope("dis"):
            _, l_x_tilde = self._discriminator(self.x_tilde)

        with tf.variable_scope("gen", reuse=True):
            self.x_p = self._generator(self.z_p)

        with tf.variable_scope("dis", reuse=True):
            self.d_x, l_x = self._discriminator(self.x)  # positive examples

        with tf.variable_scope("dis", reuse=True):
            self.d_x_p, _ = self._discriminator(self.x_p)

        with tf.variable_scope("loss"):
            SSE_loss = tf.reduce_mean(tf.square(self.x - self.x_tilde))  # This is what a normal VAE uses

            # We clip gradients of KL divergence to prevent NANs
            self.KL_loss = tf.reduce_sum(-0.5 * tf.reduce_sum(1 + tf.clip_by_value(self.z_x_log_sigma_sq, -10.0, 10.0)
                                           - tf.square(tf.clip_by_value(self.z_x_mean, -10.0, 10.0))
                                           - tf.exp(tf.clip_by_value(self.z_x_log_sigma_sq, -10.0, 10.0)), 1))/(self.image_size * self.image_size)

            # Discriminator Loss
            self.D_loss = tf.reduce_mean(-1. * (tf.log(tf.clip_by_value(self.d_x, 1e-5, 1.0)) + tf.log(tf.clip_by_value(1.0 - self.d_x_p, 1e-5, 1.0))))

            # Generator Loss
            self.G_loss = tf.reduce_mean(-1. * (tf.log(tf.clip_by_value(self.d_x_p, 1e-5, 1.0))))

            # Lth Layer Loss - the 'learned similarity measure'
            self.LL_loss = tf.reduce_sum(tf.square(l_x - l_x_tilde))/(self.image_size * self.image_size)

            # summary
            with tf.name_scope("loss_summary"):
                tf.summary.histogram("LL_loss", self.LL_loss)

        with tf.variable_scope("optimizer"):

            # specify loss to parameters
            params = tf.trainable_variables()
            E_params = [i for i in params if 'enc' in i.name]
            G_params = [i for i in params if 'gen' in i.name]
            D_params = [i for i in params if 'dis' in i.name]

            # Calculate the losses specific to encoder, generator, decoder
            L_e = tf.clip_by_value(self.KL_loss + self.LL_loss, -100, 100)
            L_g = tf.clip_by_value(self.LL_loss + self.G_loss, -100, 100)
            L_d = tf.clip_by_value(self.D_loss, -100, 100)

            optimizer_E = tf.train.AdamOptimizer(self.lr_E, epsilon=1.0)
            grads = optimizer_E.compute_gradients(L_e, var_list=E_params)
            self.train_E = optimizer_E.apply_gradients(grads, global_step=self.global_step)

            optimizer_G = tf.train.AdamOptimizer(self.lr_G, epsilon=1.0)
            grads = optimizer_G.compute_gradients(L_g, var_list=G_params)
            self.train_G = optimizer_G.apply_gradients(grads, global_step=self.global_step)

            optimizer_D = tf.train.AdamOptimizer(self.lr_D, epsilon=1.0)
            grads = optimizer_D.compute_gradients(L_d, var_list=D_params)
            self.train_D = optimizer_D.apply_gradients(grads, global_step=self.global_step)

        # check tensors
        self._check_tensors()

        # init session
        self.sess = tf.Session()

        self.merged_summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.experiment_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _sigmoid(self, x, shift, mult):
        """
        Using this sigmoid to discourage one network overpowering the other
        """
        return 1 / (1 + np.exp(-(x + shift) * mult))

    def _encoder(self, input_tensor, output_size):
        net = tf.reshape(input_tensor, [self.batch_size, self.image_size, self.image_size, self.channels])
        net1 = layers.conv2d(net, 64, kernel_size=5, stride=2, activation_fn=model_ops.leaky_relu_batch_norm)
        net2 = layers.conv2d(net1, 128, kernel_size=5, stride=2, activation_fn=model_ops.leaky_relu_batch_norm)
        net3 = layers.conv2d(net2, 256, kernel_size=5, stride=2, padding='VALID', activation_fn=None)
        flat = layers.flatten(net3)
        fc = layers.fully_connected(flat, 2 * output_size)
        z_mean = fc[:, :output_size]
        z_log_sigma_q = fc[:, output_size:]

        # summary
        with tf.name_scope("encoder_summary"):
            tf.summary.image("net1",  tf.expand_dims(tf.reduce_mean(net1, axis=-1), axis=-1))

        return z_mean, z_log_sigma_q

    def _generator(self, input_tensor):
        fc = model_ops.linear_contrib(input_tensor, 8 * 8 * 256, activation_fn=None)
        z = tf.reshape(fc, shape=(tf.shape(input_tensor)[0], 8, 8, 256))
        net = layers.conv2d_transpose(z, 256, kernel_size=8, stride=2, activation_fn=model_ops.leaky_relu_batch_norm)
        net = layers.conv2d_transpose(net, 128, kernel_size=5, stride=2, activation_fn=model_ops.leaky_relu_batch_norm)
        net = layers.conv2d_transpose(net, 32, kernel_size=5, stride=2, activation_fn=model_ops.leaky_relu_batch_norm)
        net = layers.conv2d_transpose(net, self.channels, kernel_size=5, stride=1, activation_fn=None)
        net = tf.nn.sigmoid(net)
        return net

    def _discriminator(self, input_tensor):
        net = tf.reshape(input_tensor, [self.batch_size, self.image_size, self.image_size, self.channels])
        net = layers.conv2d(net, 32, kernel_size=5, stride=1, activation_fn=model_ops.leaky_relu_batch_norm)
        net = layers.conv2d(net, 128, kernel_size=5, stride=2, activation_fn=model_ops.leaky_relu_batch_norm)
        net = layers.conv2d(net, 256, kernel_size=5, stride=2, padding='VALID', activation_fn=model_ops.leaky_relu_batch_norm)
        lth_layer = layers.fully_connected(net, 1024, activation_fn=None)
        D = layers.fully_connected(lth_layer, 1, activation_fn=tf.nn.sigmoid)
        return D, lth_layer

    def _check_tensors(self):
        if tf.trainable_variables():
            for v in tf.trainable_variables():
                print("%s : %s" % (v.name, v.get_shape()))

    def update_params(self, x):
        e_current_lr = self.e_learning_rate * self._sigmoid(0, -.5, 15)
        g_current_lr = self.g_learning_rate * self._sigmoid(0, -.5, 15)
        d_current_lr = self.d_learning_rate * self._sigmoid(0, -.5, 15)

        _, _, _, D_err, G_err, KL_err, LL_err, self.d_fake, self.d_real = self.sess.run([
            self.train_E, self.train_G, self.train_D,
            self.D_loss, self.G_loss, self.KL_loss, self.LL_loss,
            self.d_x_p, self.d_x], feed_dict={self.x: x,
                                              self.lr_E: e_current_lr,
                                              self.lr_G: g_current_lr,
                                              self.lr_D: d_current_lr})

        # summary update
        summary = self.sess.run(self.merged_summary_op, feed_dict={self.x: x})
        self.summary_writer.add_summary(summary, global_step=tf.train.global_step(self.sess, self.global_step))

        return D_err, G_err, KL_err, LL_err, self.d_fake, self.d_real

    def generate_samples(self, sess, num_samples):
        z = np.random.normal(size=(num_samples, self.hidden_size))
        samples = sess.run(self.x_p, feed_dict={self.z_p: z})
        return np.array(samples)

    def loss_msg(self, x, epoch):
        D_err, G_err, KL_err, LL_err = self.sess.run([self.D_loss, self.G_loss, self.KL_loss, self.LL_loss], feed_dict={self.x: x})
        return "epoch: %3d," % epoch + " discriminator loss: %.6f," % D_err + " generator loss: %.6f," % G_err + " KL loss: %.6f," % KL_err + " LL loss: %.6f," % LL_err
