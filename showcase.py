import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import color
from colorizer import Colorizer, DatasetIterator
import cv2

if __name__ == "__main__":
    """
    OLD:
    real    1m14.982s
    user    3m10.296s
    NEW:
    real    0m47.679s
    user    2m11.769s
    """
    orignal_image_folder = 'datasets/preprocessed_1000'

    filenames = [os.path.join(orignal_image_folder, record) for record in os.listdir(
        orignal_image_folder) if os.path.isfile(os.path.join(orignal_image_folder, record))]

    n_epochs = 1
    batch_size = 15

    iterator = DatasetIterator(filenames, n_epochs, batch_size, shuffle=True)
    colorizer = Colorizer(iterator)
    new_image_node, example_node = colorizer.showcase()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'models/my-model')

        new_ab_list, example = sess.run([new_image_node, example_node])

        new_ab_list = colorizer.bin_prob_to_ab(new_ab_list)

        l_channel_list = example['l_channel'].values.reshape(
            (batch_size, 128, 128, 1))
        labels_list = example['labels'].values.reshape(
            (batch_size, 128, 128, 1))
        labels_list = colorizer.bin_to_ab(labels_list)

        for i in range(batch_size):

            new_ab = new_ab_list[i]

            l_channel = l_channel_list[i]
            real_ab = labels_list[i]

            l_channel = (l_channel + 1)*100/2

            lab_image = np.concatenate((l_channel, real_ab), axis=2)
            real_image = color.lab2rgb(lab_image)

            new_lab_image = np.concatenate((l_channel, new_ab), axis=2)
            new_image = color.lab2rgb(new_lab_image)

            real_image = np.clip(cv2.resize(real_image, dsize=(
                299, 299), interpolation=cv2.INTER_CUBIC), 0, 1)
            new_image = np.clip(cv2.resize(new_image, dsize=(
                299, 299), interpolation=cv2.INTER_CUBIC), 0, 1)

            print(np.max(real_image))
            print(np.min(real_image))

            print(np.max(new_image))
            print(np.min(new_image))

            plt.subplot(131)
            plt.imshow(real_image)

            plt.subplot(132)
            plt.imshow(new_image)

            plt.subplot(133)
            plt.imshow(l_channel.reshape((128, 128)), cmap='gray')

            plt.show()
