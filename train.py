import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from colorizer import Colorizer, DatasetIterator

if __name__ == "__main__":
    orignal_image_folder = 'datasets/preprocessed'

    filenames = [os.path.join(orignal_image_folder, record) for record in os.listdir(
        orignal_image_folder) if os.path.isfile(os.path.join(orignal_image_folder, record))]

    n_epochs = 1000
    batch_size = 1

    iterator = DatasetIterator(filenames, n_epochs, batch_size)
    colorizer = Colorizer(iterator)
    optimizer, loss_node = colorizer.training_op()

    saver = tf.train.Saver()

    load = True
    save = True

    count = 0
    losses = []
    with tf.Session() as sess:
        if load:
            saver.restore(sess, 'models/my-model')
        else:
            sess.run(tf.global_variables_initializer())
        try:
            while True:
                _, loss = sess.run(
                    [optimizer, loss_node])
                print(loss)
                losses.append(loss)
        except tf.errors.OutOfRangeError:
            print("Done with training!")

        if save:
            saver.save(sess, 'models/my-model')
            print("model saved!")

        plt.plot(losses)
        plt.show()
