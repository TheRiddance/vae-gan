""" VAE """

import os
import tensorflow as tf

import model_ops
from visualizer import ImageVisualizer


class VAE:

    def __init__(self, batch_size, hidden_size=128, learning_rate=1e-3, image_size=64):
        self.image_size = image_size
        self.input_tensor = tf.placeholder(tf.float32, [None, image_size * image_size])

        with tf.variable_scope("model"):
            encoded = self._encoder(self.input_tensor, hidden_size * 2, batch_size=batch_size, image_size=image_size)

            input_mean = encoded[:, :hidden_size]
            input_stddev = tf.sqrt(tf.exp(encoded[:, hidden_size:]))

            epsilon = tf.random_normal([batch_size, hidden_size])
            input_sample = tf.add(input_mean, tf.multiply(tf.sqrt(tf.exp(input_stddev)), epsilon))

            self.output_reconstr = self._decoder(input_sample, batch_size)

        with tf.variable_scope("model", reuse=True):
            self.sampled_tensor = self._decoder(tf.random_normal([batch_size, hidden_size]), batch_size)

        with tf.variable_scope("loss"):
            self.Lx = self._reconstruction_loss(self.output_reconstr, self.input_tensor)
            self.Lz = self._latent_loss(input_mean, input_stddev)

            self.loss = self.Lx + self.Lz

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _encoder(self, input_tensor, output_size, batch_size, image_size=64):

        net = tf.reshape(input_tensor, [batch_size, image_size, image_size, 1])
        net = model_ops.conv2d(net, 32, stride=2, name="enc_conv1")
        net = model_ops.conv2d(net, 64, stride=2, name="enc_conv2")
        net = model_ops.conv2d(net, 128, stride=2, padding='VALID', name="enc_conv3")
        net = tf.nn.dropout(net, keep_prob=0.9)
        net = tf.reshape(net, [batch_size, -1])
        return model_ops.linear(net, output_size, "enc_fully")

    def _decoder(self, input_tensor, batch_size):

        net = tf.expand_dims(input_tensor, 1)
        net = tf.expand_dims(net, 1)
        net = model_ops.conv2d_back(net, [batch_size, 8, 8, 128], kernel=8, padding='VALID', name="dec_conv1")
        net = model_ops.conv2d_back(net, [batch_size, 16, 16, 64], stride=2, name="dec_conv2")
        net = model_ops.conv2d_back(net, [batch_size, 32, 32, 32], stride=2, name="dec_conv3")
        net = model_ops.conv2d_back(net, [batch_size, 64, 64, 1], stride=2, name="dec_conv4")
        return tf.reshape(net, [batch_size, -1])

    def _reconstruction_loss(self, reconstr_tensor, target_tensor, epsilon=1e-8):
        """
            maximum likelihood estimation
        """
        #return -tf.reduce_sum(target_tensor * tf.log(epsilon + reconstr_tensor) + (1 - target_tensor) * tf.log(epsilon + 1 - reconstr_tensor))
        return -tf.reduce_sum(target_tensor * tf.log(reconstr_tensor + epsilon) + (1.0 - target_tensor) * tf.log(1.0 - reconstr_tensor + epsilon))

    def _latent_loss(self, z_mean, z_stddev, epsilon=1e-8):
        #return -0.5 * tf.reduce_sum(1 + z_stddev - tf.square(z_mean) - tf.exp(z_stddev))
        return tf.reduce_sum(0.5 * (tf.square(z_mean) + tf.square(z_stddev) - 2.0 * tf.log(z_stddev + epsilon) - 1.0))

    def update_params(self, input_tensor):
        _, loss, Lx, Lz = self.sess.run([self.optimizer, self.loss, self.Lx, self.Lz], feed_dict={self.input_tensor: input_tensor})
        return loss, Lx, Lz

    def generate_and_save_images(self, num_samples, directory, epoch, images):
        # create experiment folder
        experiment_dir = os.path.join(directory, "VAE")
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
            print('created directory:', experiment_dir)

        generated_samples = self.sess.run(self.sampled_tensor)
        visualizer = ImageVisualizer(experiment_dir, image_size=self.image_size)
        visualizer.save_generated_samples(generated_samples, epoch)

    def print_loss(self, input_tensor, epoch):
        loss, Lx, Lz = self.sess.run([self.loss, self.Lx, self.Lz], feed_dict={self.input_tensor: input_tensor})
        print("epoch: %3d" % epoch, "loss %.2f" % loss, "Lx %.2f" % Lx, "Lz %.2f" % Lz)
