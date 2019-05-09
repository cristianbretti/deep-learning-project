import numpy as np
import tensorflow as tf
from skimage import color

from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer
from tensorflow.keras.models import Sequential


class Colorizer():
    def __init__(self, iterator, learning_rate):
        self.iterator = iterator
        self.batch_size = self.iterator.batch_size
        self.learning_rate = learning_rate

        self.resnet_preprocessed_shape = (self.batch_size, 8, 8, 1536)
        self.out_network_shape = (self.batch_size, 299, 299, 2)
        self.out_image_size = (299, 299)  # UPSAMPLE

        self._upscale = self._build_upscale()

    def _build_upscale(self):
        """
        Upscale Network architecture
        input   = (8, 8, 1536)
        layer_1 = (8, 8, 256)
        layer_2 = (8, 8, 128)
        layer_3 = (32, 32, 128)   # UPSAMPLE
        layer_4 = (32, 32, 64)
        layer_5 = (32, 32, 32)
        layer_6 = (128, 128, 32)  # UPSAMPLE
        layer_7 = (128, 128, 8)
        layer_8 = (256, 256, 8)   # UPSAMPLE
        layer_9 = (256, 256, 2)
        """
        model = Sequential(name='upscale_model')
        model.add(InputLayer(batch_input_shape=self.resnet_preprocessed_shape))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((4, 4)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((4, 4)))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(2, (3, 3), activation='relu', padding='same'))
        return model

    def upscale(self, predicted):
        predicted = tf.reshape(predicted, self.resnet_preprocessed_shape)
        predicted = tf.nn.relu(predicted)
        upscaled = self._upscale(predicted)
        new_image = tf.image.resize_nearest_neighbor(
            upscaled, self.out_image_size)
        return new_image

    def loss_function(self, example):
        new_image = self.upscale(example['predicted'].values)
        ab_channels_real = example['ab_channels'].values
        ab_channels_real = tf.reshape(ab_channels_real, self.out_network_shape)
        loss = tf.reduce_mean(
            tf.squared_difference(
                ab_channels_real, new_image))
        return loss

    def training_op(self):
        next_example = self.iterator.get_next()
        loss = self.loss_function(next_example)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(loss)
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
