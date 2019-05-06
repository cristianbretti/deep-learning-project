import numpy as np
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


# For reading files
sess = tf.Session()
filename = "datasets/preprocessed/done-00000-of-00004.tfrecord"

filenames = tf.placeholder(tf.string, shape=[None])
filenames = [filename]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(parser)

dataset = dataset.batch(1)
iterator = dataset.make_one_shot_iterator()

next_example = iterator.get_next()

test = sess.run(next_example)

l_channel = test['l_channel'].values.reshape((299, 299, 1))
ab_channels = test['ab_channels'].values.reshape((299, 299, 2))

l_channel = (l_channel + 1)*100/2
ab_channels = (ab_channels)*127

lab_image = np.concatenate((l_channel, ab_channels), axis=2)
image = color.lab2rgb(lab_image)

plt.imshow(image)
plt.show()
