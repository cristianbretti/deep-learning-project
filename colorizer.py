import numpy as np
import tensorflow as tf
from skimage import color

from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer, BatchNormalization, Dropout, Activation, Softmax
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential


class Colorizer():
    def __init__(self, iterator, learning_rate=0.001):
        self.iterator = iterator
        self.batch_size = self.iterator.batch_size
        self.learning_rate = learning_rate

        self.n_classes = 289

        self.resnet_preprocessed_shape = (self.batch_size, 8, 8, 1536)
        self.labels_shape = (self.batch_size, 128, 128)
        self.out_network_shape = (self.batch_size, 128, 128, self.n_classes)

        self._upscale = self._build_upscale()

    def set_iterator(self, iterator):
        self.iterator = iterator
        self.batch_size = self.iterator.batch_size

    def _build_upscale(self):
        """
        Upscale Network architecture
        input   = (8, 8, 1536)
        layer_1 = (32, 32, 768)
        layer_2 = (64, 64, 384)
        layer_3 = (128, 128, 289)
        """
        model = Sequential(name='upscale_model')
        model.add(InputLayer(batch_input_shape=self.resnet_preprocessed_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Dropout(0.2))

        model.add(Conv2D(768, (3, 3), activation='relu',
                         padding='same', kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization(axis=3))
        model.add(Dropout(0.2))
        model.add(UpSampling2D((4, 4)))

        model.add(Conv2D(384, (3, 3), activation='relu',
                         padding='same', kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization(axis=3))
        model.add(Dropout(0.2))
        model.add(UpSampling2D((2, 2)))

        model.add(Conv2D(self.n_classes, (3, 3), padding='same',
                         kernel_regularizer=l2(0.01)))
        model.add(UpSampling2D((2, 2)))

        return model

    def loss_function(self, predicted, labels):
        new_image = self._upscale(predicted)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels, new_image))
        return loss

    def prepare_next_data_batch(self):
        example = self.iterator.get_next()
        predicted = example['predicted'].values
        predicted = tf.reshape(predicted, self.resnet_preprocessed_shape)

        labels = example['labels'].values
        labels = tf.reshape(labels, self.labels_shape)
        labels = tf.cast(labels, tf.int32)
        one_hot = tf.one_hot(labels, self.n_classes)
        return predicted, one_hot

    def training_op(self):
        predicted, one_hot_labels = self.prepare_next_data_batch()
        loss = self.loss_function(predicted, one_hot_labels)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(loss)
        return optimizer, loss

    def showcase(self):
        example = self.iterator.get_next()
        predicted = example['predicted'].values
        predicted = tf.reshape(predicted, self.resnet_preprocessed_shape)
        new_image = self._upscale(predicted)
        new_image = tf.nn.softmax(new_image)
        return new_image, example

    def ab_to_bin(self, ab_channels):
        # set values between 0 and 254
        ab_channels = ab_channels * 127 + 127
        ab_channels = np.floor(ab_channels / 15)
        labels = ab_channels[:, :, :, 0] * 17 + ab_channels[:, :, :, 1]
        labels = labels.reshape(
            labels.shape[0], labels.shape[1], labels.shape[2], 1)
        return labels

    def bin_prob_to_ab(self, bins):
        # bins = (batch, 299, 299, 289)

        bins = np.argmax(bins, axis=3).reshape(
            bins.shape[0], bins.shape[1], bins.shape[2], 1)

        return self.bin_to_ab(bins)

    def bin_to_ab(self, bins):
        a_values = np.floor(bins / 17) * 15
        b_values = np.mod(bins, 17) * 15

        ab_channels = np.concatenate((a_values, b_values), axis=3)
        ab_channels = ab_channels - 127
        return ab_channels


class DatasetIterator():
    def __init__(self, filenames, batch_size=100, shuffle=True):
        self.keys_to_features = {
            'filename': tf.FixedLenFeature((), tf.string, default_value=""),
            'l_channel': tf.VarLenFeature(dtype=tf.float32),
            'labels': tf.VarLenFeature(dtype=tf.float32),
            'predicted': tf.VarLenFeature(dtype=tf.float32),
        }
        self.filenames = filenames
        self.batch_size = batch_size

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parser)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        self.dataset = dataset.batch(self.batch_size)
        self.iterator = self.dataset.make_initializable_iterator()

    def get_next(self):
        return self.iterator.get_next()

    def initializer(self):
        return self.iterator.initializer

    def parser(self, record):
        return tf.parse_single_example(record, self.keys_to_features)
