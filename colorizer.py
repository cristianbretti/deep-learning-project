import numpy as np
import tensorflow as tf
from skimage import color

from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential


class Colorizer():
    def __init__(self, iterator, learning_rate=0.001):
        self.iterator = iterator
        self.batch_size = self.iterator.batch_size
        self.learning_rate = learning_rate

        self.resnet_preprocessed_shape = (self.batch_size, 8, 8, 1536)
        self.out_network_shape = (self.batch_size, 299, 299, 2)
        self.out_image_size = (299, 299)

        self.predicted_input = tf.placeholder(
            dtype=tf.float32, shape=self.resnet_preprocessed_shape)
        self.labels = tf.placeholder(
            dtype=tf.float32, shape=self.out_network_shape)

        self._upscale = self._build_upscale()

    def _build_upscale(self):
        """
        Upscale Network architecture
        input   = (8, 8, 1536)
        layer_1 = (16, 16, 768)
        layer_2 = (32, 32, 384)
        layer_3 = (64, 64, 289)
        """
        model = Sequential(name='upscale_model')
        model.add(InputLayer(batch_input_shape=self.resnet_preprocessed_shape))
        model.add(Dropout(0.2))
        model.add(BatchNormalization(axis=3))
        # model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization(axis=3))
        model.add(UpSampling2D((4, 4)))
        # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization(axis=3))
        # model.add(UpSampling2D((4, 4)))
        # model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((8, 8)))
        model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
        return model

    def upscale(self, predicted):
        predicted = tf.nn.relu(predicted)
        upscaled = self._upscale(predicted)
        new_image = tf.image.resize_nearest_neighbor(
            upscaled, self.out_image_size)
        return new_image

    def loss_function(self, predicted, labels):
        new_image = self.upscale(predicted)
        loss = tf.reduce_mean(
            tf.squared_difference(
                labels, new_image))
        return loss

    def prepare_next_data_batch(self):
        example = self.iterator.get_next()
        predicted = example['predicted'].values
        predicted = tf.reshape(predicted, self.resnet_preprocessed_shape)

        ab_channels_real = example['ab_channels'].values
        ab_channels_real = tf.reshape(
            ab_channels_real, self.out_network_shape)
        return predicted, ab_channels_real

    def training_op(self):
        loss = self.loss_function(
            self.predicted_input, self.labels)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(loss)
        return optimizer, loss

    def showcase(self):
        example = self.iterator.get_next()
        predicted = example['predicted'].values
        predicted = tf.reshape(predicted, self.resnet_preprocessed_shape)
        new_image = self.upscale(predicted)
        return new_image, example

    def ab_to_bin(self, ab_channels):
        ab_channels = np.floor(ab_channels / 15)
        labels = ab_channels[:, :, :, 0] * 17 + ab_channels[:, :, :, 1]
        return labels.reshape(labels.shape[0], labels.shape[1], labels.shape[2], 1)

    def bin_to_ab(self, bins):
        # (batch, 299, 299, 1)
        a_values = np.floor(bins / 17) * 15
        b_values = np.mod(bins, 17) * 15

        ab_channels = np.concatenate((a_values, b_values), axis=3)
        ab_channels = ab_channels - 127
        return ab_channels


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
