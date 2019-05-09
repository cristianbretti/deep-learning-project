import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import color
from colorizer import Colorizer, DatasetIterator

if __name__ == "__main__":
    orignal_image_folder = 'datasets/preprocessed_fish'

    filenames = [os.path.join(orignal_image_folder, record) for record in os.listdir(
        orignal_image_folder) if os.path.isfile(os.path.join(orignal_image_folder, record))]

    n_epochs = 1
    batch_size = 1

    iterator = DatasetIterator(filenames, n_epochs, batch_size, shuffle=False)
    colorizer = Colorizer(iterator)
    new_image_node, example_node = colorizer.showcase()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'models/my-model')

        new_ab_list, example = sess.run([new_image_node, example_node])

        l_channel_list = example['l_channel'].values.reshape(
            (batch_size, 299, 299, 1))
        real_ab_list = example['ab_channels'].values.reshape(
            (batch_size, 299, 299, 2))

        for i in range(batch_size):

            new_ab = new_ab_list[i].reshape((299, 299, 2))
            new_ab = (new_ab)*127
            print(new_ab)

            l_channel = l_channel_list[i]
            real_ab = real_ab_list[i]
            l_channel = (l_channel + 1)*100/2
            real_ab = (real_ab)*127

            lab_image = np.concatenate((l_channel, real_ab), axis=2)
            real_image = color.lab2rgb(lab_image)

            new_lab_image = np.concatenate((l_channel, new_ab), axis=2)
            new_image = color.lab2rgb(new_lab_image)

            plt.subplot(131)
            plt.imshow(real_image)

            plt.subplot(132)
            plt.imshow(new_image)

            plt.subplot(133)
            plt.imshow(l_channel.reshape((299, 299)), cmap='gray')

            plt.show()
