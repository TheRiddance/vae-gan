"""
    DRAW model visualizer
"""

import os
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt


class ImageVisualizer:
    def __init__(self, experiment_dir, image_size):
        self.ROWS = 10
        self.COLS = 20
        self.NUM_SAMPLES = self.ROWS * self.COLS

        self.experiment_dir = experiment_dir
        self.image_size = image_size

    def training_data_sample(self, train_data, save_name="dataset_samples.png"):
        idxs = np.random.randint(0, train_data.length(), self.NUM_SAMPLES)
        rand_imgs = train_data.images[idxs]
        image_matrix = self._reconstr_grid(rand_imgs, self.NUM_SAMPLES, self.image_size)
        # save image matrix
        self._save_image_matrix(image_matrix, save_name)

    def save_generated_samples(self, sample_images, epoch):
        # sample reconstruction
        image_matrix = self._reconstr_grid(sample_images, sample_images.shape[0], self.image_size)
        name = "epoch_%03d.png" % epoch
        self._save_image_matrix(image_matrix, name)

        print("generated and saved images...")

    def _reconstr_grid(self, x, num_samples, image_size):
        """
            creates image matrix with:
            height: self.ROWS
            width: self.COLS
            input shape must be compatible with
        """
        imgs = np.reshape(x, (num_samples, image_size, image_size))

        image_matrix = np.ones((self.ROWS * image_size, self.COLS * image_size))
        for i in range(0, self.ROWS):
            for j in range(0, self.COLS):
                if i * self.COLS + j < num_samples:
                    image_matrix[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size] = imgs[i * self.COLS + j][0:image_size][0:image_size]

        return image_matrix

    def _save_image_matrix(self, image_matrix, save_name):
        """
            save images to current experiment directory
        """
        save_path = os.path.join(self.experiment_dir, save_name)
        imsave(save_path, image_matrix)


class ReconstructionVisualizer:
    def __init__(self, experiment_dir, image_size):
        self.ROWS = 3
        self.COLS = 8
        self.NUM_SAMPLES = self.COLS

        self.experiment_dir = experiment_dir
        self.image_size = image_size

    def save_generated_samples(self, random_samples, reconstructed_samples, examples, epoch):

        fig, ax = plt.subplots(nrows=3, ncols=self.NUM_SAMPLES, figsize=(18, 6))
        for i in range(self.NUM_SAMPLES):
            ax[(0, i)].imshow(self._create_image(random_samples[i]), cmap=plt.cm.gray, interpolation='nearest')
            ax[(1, i)].imshow(self._create_image(reconstructed_samples[i]), cmap=plt.cm.gray, interpolation='nearest')
            ax[(2, i)].imshow(self._create_image(examples[i]), cmap=plt.cm.gray, interpolation='nearest')
            ax[(0, i)].axis('off')
            ax[(1, i)].axis('off')
            ax[(2, i)].axis('off')
        fig.suptitle('Top: random points in z space | Bottom: inputs | Middle: reconstructions')
        plt.savefig(os.path.join(self.experiment_dir, "reonstruction_" + str(epoch) + ".png"))

        print("generated and saved images...")

    def _create_image(self, im):
        return np.reshape(im, (self.image_size, self.image_size))
