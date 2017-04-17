""" Training """

import tensorflow as tf

from vae import VAE
from gan import GAN
from vae_gan import VAE_GAN
import hdf5_dataset
from checkpoint_saver import CheckpointSaver
from visualizer import ImageVisualizer

flags = tf.app.flags
flags.DEFINE_string("model", "GAN", "choose a model: GAN or VAE")
flags.DEFINE_string("dataset", "datasets/samples.h5", "sample data dir")
flags.DEFINE_string("data_dir", "results/", "checkpoint and logging results dir")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("image_size", 64, "image size")
flags.DEFINE_integer("channels", 3, "color channels")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_integer("hidden_size", 515, "size of latent (feature?) space")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_integer("generation_step", 1, "generate random images")
FLAGS = flags.FLAGS


def main(_):
    # create checkpoint saver
    # the checkpoint saver, can create checkpoint files, which later can be use to restore a model state, but it also
    # audits the model progress to a log file
    checkpoint_saver = CheckpointSaver(FLAGS.data_dir, experiment_name=FLAGS.model)
    checkpoint_saver.save_experiment_config(FLAGS.__dict__['__flags'])

    train_data = hdf5_dataset.read_data_set(FLAGS.dataset, image_size=FLAGS.image_size, shape=(FLAGS.image_size, FLAGS.image_size, FLAGS.channels), binarized=True, validation=0).train

    # create a data visualizer
    visualizer = ImageVisualizer(checkpoint_saver.get_experiment_dir(), image_size=FLAGS.image_size)
    visualizer.training_data_sample(train_data)

    if FLAGS.model == "VAE":
        model = VAE(FLAGS.batch_size, FLAGS.hidden_size, FLAGS.learning_rate, image_size=FLAGS.image_size)
    elif FLAGS.model == "GAN":
        model = GAN(FLAGS.batch_size, FLAGS.hidden_size, FLAGS.learning_rate, image_size=FLAGS.image_size)
    elif FLAGS.model == "VAE-GAN":
        model = VAE_GAN(FLAGS.hidden_size, image_size=FLAGS.image_size, channels=FLAGS.channels, experiment_name=checkpoint_saver.get_experiment_dir())
    else:
        print("this model is not supported")
        return

    print("start", type(model).__name__, "model")

    for epoch in range(FLAGS.max_epoch):

        for images in train_data.next_batch(FLAGS.batch_size):
            model.update_params(images)

        checkpoint_saver.audit_loss(model.loss_msg(images, epoch))

        if epoch % FLAGS.generation_step == 0:
            visualizer.save_generated_samples(model.generate_samples(model.sess, num_samples=200), epoch)

if __name__ == '__main__':
    tf.app.run()
