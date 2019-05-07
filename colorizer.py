import numpy as np
import tensorflow as tf
from skimage import color


class Colorizer():
    def __init__(self, iterator):
        self.iterator = iterator
        self.batch_size = self.iterator.batch_size

        self.resnet_preprocessed_shape = (self.batch_size, 8, 8, 1536)
        self.output_layer_shape_1 = (self.batch_size, 8, 8, 256)
        self.output_layer_shape_2 = (self.batch_size, 8, 8, 128)
        self.output_layer_shape_3 = (32, 32)  # UPSAMPLE
        self.output_layer_shape_4 = (self.batch_size, 32, 32, 64)
        self.output_layer_shape_5 = (self.batch_size, 32, 32, 32)
        self.output_layer_shape_6 = (128, 128)  # UPSAMPLE
        self.output_layer_shape_7 = (self.batch_size, 128, 128, 8)
        self.output_layer_shape_8 = (299, 299)  # UPSAMPLE
        self.output_layer_shape_9 = (self.batch_size, 299, 299, 2)

        self.filter_conv_1 = tf.Variable(tf.random_normal(
            [3, 3, 1536, 256], stddev=np.sqrt(2 / (1536 + 256))),  name="filter_conv_1")
        self.filter_conv_2 = tf.Variable(tf.random_normal(
            [3, 3, 256, 128], stddev=np.sqrt(2 / (128 + 256))),  name="filter_conv_2")
        self.filter_conv_4 = tf.Variable(
            tf.random_normal([3, 3, 128, 64], stddev=np.sqrt(
                2 / (128 + 64))),  name="filter_conv_4")
        self.filter_conv_5 = tf.Variable(
            tf.random_normal([3, 3, 64, 32], stddev=np.sqrt(
                2 / (64 + 32))),  name="filter_conv_5")
        self.filter_conv_7 = tf.Variable(
            tf.random_normal([3, 3, 32, 8], stddev=np.sqrt(
                2 / (32 + 8))),  name="filter_conv_7")
        self.filter_conv_9 = tf.Variable(
            tf.random_normal([3, 3, 8, 2], stddev=np.sqrt(
                2 / (8 + 2))),  name="filter_conv_9")

    def upscale(self, predicted):
        predicted = tf.reshape(predicted, self.resnet_preprocessed_shape)
        predicted = tf.nn.leaky_relu(predicted)
        # (8, 8, 1536)
        layer_1 = tf.nn.conv2d(
            predicted, self.filter_conv_1, strides=[1, 1, 1, 1], padding='SAME')
        layer_1 = tf.nn.leaky_relu(layer_1)
        # (8, 8, 256)
        layer_2 = tf.nn.conv2d(
            layer_1, self.filter_conv_2, strides=[1, 1, 1, 1], padding='SAME')
        layer_2 = tf.nn.leaky_relu(layer_2)
        # (8, 8, 128)
        layer_3 = tf.image.resize_nearest_neighbor(
            layer_2, self.output_layer_shape_3)
        # (32, 32, 128)
        layer_4 = tf.nn.conv2d(
            layer_3, self.filter_conv_4, strides=[1, 1, 1, 1], padding='SAME')
        layer_4 = tf.nn.leaky_relu(layer_4)
        # (32, 32, 64)
        layer_5 = tf.nn.conv2d(
            layer_4, self.filter_conv_5, strides=[1, 1, 1, 1], padding='SAME')
        layer_5 = tf.nn.leaky_relu(layer_5)
        # (32, 32, 32)
        layer_6 = tf.image.resize_nearest_neighbor(
            layer_5, self.output_layer_shape_6)
        # (128, 128, 32)
        layer_7 = tf.nn.conv2d(
            layer_6, self.filter_conv_7, strides=[1, 1, 1, 1], padding='SAME')
        layer_7 = tf.nn.leaky_relu(layer_7)
        # (128, 128, 8)
        layer_8 = tf.image.resize_nearest_neighbor(
            layer_7, self.output_layer_shape_8)
        # (299, 299, 8)
        layer_9 = tf.nn.conv2d(
            layer_8, self.filter_conv_9, strides=[1, 1, 1, 1], padding='SAME')
        layer_9 = tf.nn.tanh(layer_9)
        # (299, 299, 2)
        return layer_9

    def loss_function(self, example):
        new_image = self.upscale(example['predicted'].values)
        ab_channels_real = example['ab_channels'].values
        ab_channels_real = tf.reshape(
            ab_channels_real, self.output_layer_shape_9)
        # loss = tf.losses.mean_squared_error(
        loss = tf.reduce_mean(
            tf.squared_difference(
                ab_channels_real, new_image))
        return loss

    def training_op(self):
        next_example = self.iterator.get_next()
        loss = self.loss_function(next_example)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
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
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.batch_size)
        self.dataset = dataset.repeat(self.n_epochs)
        self.iterator = self.dataset.make_one_shot_iterator()

    def get_next(self):
        return self.iterator.get_next()

    def parser(self, record):
        return tf.parse_single_example(record, self.keys_to_features)
