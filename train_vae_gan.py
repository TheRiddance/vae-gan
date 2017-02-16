""" Training """

import tensorflow as tf

from vae_gan import VAE_GAN
import hdf5_dataset

flags = tf.app.flags

flags.DEFINE_string("samples_dir", "samples/", "sample data dir")
flags.DEFINE_string("data_dir", "data/", "checkpoint and logging data dir")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("image_size", 64, "image size")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_integer("hidden_size", 256, "size of the hidden VAE unit")
flags.DEFINE_integer("generation_step", 5, "generate random images")

FLAGS = flags.FLAGS


def main(_):
    train_data = hdf5_dataset.read_data_set(FLAGS.samples_dir, image_size=FLAGS.image_size, shape=FLAGS.image_size * FLAGS.image_size, binarized=True).train

    model = VAE_GAN(FLAGS.batch_size, FLAGS.hidden_size, FLAGS.learning_rate, image_size=FLAGS.image_size)

    print("start", type(model).__name__, "model")

    for epoch in range(FLAGS.max_epoch):

        for images in train_data.next_batch(FLAGS.batch_size):
            D_err, G_err, KL_err, LL_err, d_fake, d_real = model.update_params(images)

        print("epoch: %3d" % epoch, "D loss  %.4f" % D_err, "G loss  %.4f" % G_err, "KL loss  %.4f" % KL_err, "LL loss  %.4f" % LL_err)

        if epoch % FLAGS.generation_step == 0:
            model.generate_and_save_images(images, FLAGS.data_dir, epoch)

if __name__ == '__main__':
    tf.app.run()
