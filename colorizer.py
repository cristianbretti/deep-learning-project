import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import color


class Colorizer():
    def __init__(self, iterator):
        self.iterator = iterator
        self.batch_size = self.iterator.batch_size

        self.layer_filter = tf.Variable(tf.random_normal(
            [3, 3, 2, 1536]), name="layer_filter")

        self.in_shape_1 = (self.batch_size, 8, 8, 1536)
        self.out_shape_1 = (self.batch_size, 299, 299, 2)

    def upscale(self, predicted):
        predicted = tf.reshape(predicted, self.in_shape_1)
        image = tf.nn.conv2d_transpose(
            predicted, self.layer_filter, self.out_shape_1, strides=[1, 42, 42, 1])
        return image

    def loss_function(self, example):
        new_image = self.upscale(example['predicted'].values)
        ab_channels_real = example['ab_channels'].values
        ab_channels_real = tf.reshape(ab_channels_real, self.out_shape_1)
        loss = tf.losses.mean_squared_error(
            ab_channels_real, new_image)
        return loss

    def training_op(self):
        next_example = self.iterator.get_next()
        loss = self.loss_function(next_example)
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        return optimizer, loss

    def showcase(self):
        example = self.iterator.get_next()
        new_image = self.upscale(example['predicted'].values)
        return new_image, example


class DatasetIterator():
    def __init__(self, filenames, n_epochs, batch_size=100, shuffle=True):
        self.keys_to_features = {
            'filename': tf.FixedLenFeature((), tf.string, default_value=""),
            'l_channel': tf.VarLenFeature(dtype=tf.float32),
            'ab_channels': tf.VarLenFeature(dtype=tf.float32),
            'predicted': tf.VarLenFeature(dtype=tf.float32),
        }
        self.filenames = filenames
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parser)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(self.n_epochs)
        self.iterator = dataset.make_one_shot_iterator()

    def get_next(self):
        return self.iterator.get_next()

    def parser(self, record):
        return tf.parse_single_example(record, self.keys_to_features)
