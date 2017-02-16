import os
import collections
from scipy import misc
import numpy as np
import h5py

from time import time

Datasets = collections.namedtuple('Datasets', ['train', 'validation'])


# Load images from data set
# Assumes images have all same size
class HDF5_DataSet:
    def __init__(self, _hdf5_file, _images):
        self.hdf5_file = _hdf5_file
        self.images = np.array(_images)
        self.num_examples = len(self.images)
        # initial shuffle
        self.shuffle_data()

    def shuffle_data(self):
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self.images = self.images[perm]

    def next_batch(self, batch_size):
        image_batch = []
        for img in self.images:
            image_batch.append(img)
            if len(image_batch) == batch_size:
                yield np.array(image_batch)
                image_batch = []
        # all examples seen, shuffle data again<
        self.shuffle_data()

    def length(self):
        return len(self.images)


def get_normalized_image_data(image, image_size, shape):
    #TODO is this method RGB compatible?
    img = misc.imresize(image, (image_size, image_size))
    img = np.asarray(img, dtype=np.float32)
    img /= 255.0
    img = np.reshape(img, shape)
    return img


def get_binarized_image_data(image, image_size, shape):
    img = misc.imresize(image, (image_size, image_size))
    img = np.asarray(img, dtype=np.float32)
    img *= 255.0 / img.max()  # normalize between [0, 255]
    _true = img < 255.0
    _false = img == 255.0
    img[_true] = 1  # white tree
    img[_false] = 0  # black background
    return np.reshape(img, shape)


def load_dataset(hdf5_file, image_size, shape, normalization):
    res = []
    with h5py.File(hdf5_file, 'r') as _file:
        for ds in _file[_file.name]:
            res.append(normalization(_file[ds].value, image_size=image_size, shape=shape))
    return np.array(res)


def load_dataset_list(hdf5_file):
    with h5py.File(hdf5_file, 'r') as _file:
        return [ds for ds in _file[_file.name]]


def read_data_set(train_dir, image_size=64, shape=(64, 64), validation=1000, binarized=False):
    h5_file = find_file(train_dir, extensions=['.hdf5', '.h5'])
    if binarized:
        images = load_dataset(h5_file, image_size, shape, get_binarized_image_data)
    else:
        images = load_dataset(h5_file, image_size, shape, get_normalized_image_data)

    validation_images = images[:validation]
    train_images = images[validation:]

    train = HDF5_DataSet(train_dir, train_images)
    validation = HDF5_DataSet(train_dir, validation_images)

    print('data set loaded:', h5_file)
    print('image size:', image_size)
    print('shape:', shape)
    print('image binarization:', binarized)
    return Datasets(train=train, validation=validation)


def find_file(data_path, extensions):
    _file = None
    # assuming data path is a directory
    if os.path.isdir(data_path):
        for file in os.listdir(data_path):
            if file.endswith(tuple(extensions)):
                _file = os.path.join(data_path, file)
                break
        if _file:
            return _file
        else:
            print('no', extensions, 'file found in', data_path)

    # assuming data path is a file
    if os.path.isfile(data_path) and data_path.endswith(tuple(extensions)):
        return data_path
    else:
        print('No valid path or file:', data_path)
        raise IOError


def test_load_dataset():
    print('load dataset')
    start_time = time()
    data_set = read_data_set('/home/ajenal/Documents/masterthesis/project/tbone/tree_all_28k_2k_1v_skel_64x64_inverted.zip', image_size=64, shape=(64,64))
    end_time = time()
    elapsed_time = '%.3f' % (end_time - start_time)
    print('Elapsed time: ' + elapsed_time + ' seconds\n')

    print('process', data_set.train.length(), 'images')
    start_time = time()
    all_train_images = []
    for train in data_set.train.next_batch(64):
        for t in train:
            all_train_images.append(t)
    print(len(all_train_images))
    end_time = time()
    elapsed_time = '%.3f' % (end_time - start_time)
    print('Elapsed time: ' + elapsed_time + ' seconds\n')

if __name__ == '__main__':
    test_load_dataset()
    print(find_file('/home/ajenal/Documents/masterthesis/project/tbone/'))
