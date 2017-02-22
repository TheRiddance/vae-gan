import os
import collections
from scipy import misc, ndimage
import numpy as np
import h5py
from zipfile import ZipFile
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


class CelebDataset:

    def __init__(self, dataset_dir, image_size=64, channels=1):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.channels = channels

    def create_dataset_from_zip(self, path_to_zip_file, dataset_filename="celeb_dataset.h5"):
        images = []
        image_names = []
        with ZipFile(path_to_zip_file, 'r') as zfile:
            file_list = zfile.namelist()

            for img_file in file_list:
                if str(img_file).endswith('.jpg'):
                    with zfile.open(img_file) as imf:
                        img = misc.imread(imf)
                        image = self.get_normalized_image(img, self.image_size, self.image_size)
                        if self.channels == 1:
                            image = self.image2gray(image)
                        images.append(image)
                        image_names.append(img_file)

        file_name_path = os.path.join(self.dataset_dir, dataset_filename)
        with h5py.File(file_name_path, 'a') as hfile:
            self.save_images_to_hdf5(hfile, zip(image_names, images))

    def resize_width(self, image, width=64.):
        h, w = np.shape(image)[:2]
        return misc.imresize(image, [int((float(h) / w) * width), width])

    def center_crop(self, x, height=64):
        h = np.shape(x)[0]
        j = int(round((h - height) / 2.))
        return x[j:j + height, :, :]

    def get_normalized_image(self, img, width=64, height=64):
        return self.center_crop(self.resize_width(img, width=width), height=height)

    def save_images_to_hdf5(self, open_h5file, image_list):
        for img_name, img_data in image_list:
            dataset = open_h5file.create_dataset(self.get_filename(img_name), data=img_data, shape=img_data.shape)

    def get_filename(self, path):
        return os.path.splitext(os.path.basename(path))[0]

    def image2gray(self, image):
        return image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114


def test_HDF5_Dataset():
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


def test_CelebDataset():
    c = CelebDataset('/home/ajenal/', image_size=64)
    c.create_dataset_from_zip('/home/ajenal/celeb_imgs.zip')

if __name__ == '__main__':
    #test_HDF5_Dataset()
    test_CelebDataset()
