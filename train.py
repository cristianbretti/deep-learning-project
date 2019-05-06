import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import color


def parser(record):
    keys_to_features = {
        'filename': tf.FixedLenFeature((), tf.string, default_value=""),
        'l_channel': tf.VarLenFeature(dtype=tf.float32),
        'ab_channels': tf.VarLenFeature(dtype=tf.float32),
        'predicted': tf.VarLenFeature(dtype=tf.float32),
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    return parsed


def create_dataset_iterator(filename, n_epochs, batch_size):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(n_epochs)
    iterator = dataset.make_one_shot_iterator()
    return iterator


def upscale(predicted):
    predicted = tf.reshape(predicted, (batch_size, 8, 8, 1536))

    layer_filter = tf.Variable(tf.random_normal(
        [3, 3, 2, 1536]), name="layer_filter")

    image = tf.nn.conv2d_transpose(
        predicted, layer_filter, (batch_size, 299, 299, 2), strides=[1, 42, 42, 1])
    return image


def loss_function(example):
    new_image = upscale(example['predicted'].values)
    ab_channels_real = example['ab_channels'].values
    ab_channels_real = tf.reshape(ab_channels_real, (batch_size, 299, 299, 2))
    loss = tf.losses.mean_squared_error(
        ab_channels_real, new_image)
    return loss


def training_op(iterator):
    next_example = iterator.get_next()
    loss = loss_function(next_example)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    return optimizer


def showcase(filenames, n_epochs, batch_size):

    iterator = create_dataset_iterator(filenames, n_epochs, batch_size)
    example = iterator.get_next()
    new_image = upscale(example['predicted'].values)
    return new_image, example


if __name__ == "__main__":
    orignal_image_folder = 'datasets/preprocessed'

    filenames = [os.path.join(orignal_image_folder, record) for record in os.listdir(
        orignal_image_folder) if os.path.isfile(os.path.join(orignal_image_folder, record))]

    n_epochs = 1
    batch_size = 100

    iterator = create_dataset_iterator(filenames, n_epochs, batch_size)

    training_op = training_op(iterator)

    show = showcase(filenames, n_epochs, batch_size)

    train = False
    if train:
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                while True:
                    sess.run(training_op)
                    saver.save(sess, 'models/my-model')
            except tf.errors.OutOfRangeError:
                pass
    else:
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, 'models/my-model')

            show_tuple = sess.run(show)

            new_ab, example = show_tuple
            new_ab = new_ab[0]
            print(new_ab.shape)

            l_channel = example['l_channel'].values.reshape(
                (batch_size, 299, 299, 1))

            l_channel = l_channel[0]
            print(l_channel.shape)

            ab_channels = example['ab_channels'].values.reshape(
                (batch_size, 299, 299, 2))

            ab_channels = ab_channels[0]
            print(ab_channels.shape)

            l_channel = (l_channel + 1)*100/2
            ab_channels = (ab_channels)*127

            lab_image = np.concatenate((l_channel, ab_channels), axis=2)
            real_image = color.lab2rgb(lab_image)
            print(real_image.shape)

            new_ab = (new_ab)*127
            new_lab_image = np.concatenate((l_channel, new_ab), axis=2)
            new_image = color.lab2rgb(new_lab_image)
            print(new_image.shape)

            plt.imshow(real_image)
            plt.show()

            plt.imshow(new_image)
            plt.show()
