""" Training """

import tensorflow as tf

from vae import VAE
from gan import GAN
from vae_gan import VAE_GAN
import hdf5_dataset

flags = tf.app.flags

flags.DEFINE_string("model", "GAN", "choose a model: GAN or VAE")
flags.DEFINE_string("dataset", "datasets/", "sample data dir")
flags.DEFINE_string("data_dir", "data/", "checkpoint and logging data dir")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("image_size", 64, "image size")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_integer("hidden_size", 512, "size of latent (feature?) space")
flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
flags.DEFINE_integer("generation_step", 5, "generate random images")

FLAGS = flags.FLAGS


def main(_):
    train_data = hdf5_dataset.read_data_set(FLAGS.dataset, image_size=FLAGS.image_size, shape=FLAGS.image_size * FLAGS.image_size, binarized=False).train

    if FLAGS.model == "VAE":
        model = VAE(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate, image_size=FLAGS.image_size)
    elif FLAGS.model == "GAN":
        model = GAN(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate, image_size=FLAGS.image_size)
    elif FLAGS.model == "VAE-GAN":
        model = VAE_GAN(FLAGS.batch_size, FLAGS.hidden_size, FLAGS.learning_rate, FLAGS.learning_rate, FLAGS.learning_rate, image_size=FLAGS.image_size)
    else:
        print("this model is not supported")
        return

    print("start", type(model).__name__, "model")

    for epoch in range(FLAGS.max_epoch):

        for images in train_data.next_batch(FLAGS.batch_size):
            model.update_params(images)

        model.print_loss(images, epoch)

        if epoch % FLAGS.generation_step == 0:
            model.generate_and_save_images(FLAGS.batch_size, FLAGS.data_dir, epoch, images)

if __name__ == '__main__':
    tf.app.run()
