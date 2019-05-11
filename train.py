import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from colorizer import Colorizer, DatasetIterator
import time
import signal


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    if save:
        saver.save(sess, 'models/my-model')
        print("model saved!")

        losses_filename = 'models/losses'
        np.save(losses_filename, losses)
        print("Losses saved in %s" % losses_filename)

    plt.plot(losses)
    plt.show()
    exit(0)


signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    orignal_image_folder = 'datasets/preprocessed_1000'

    filenames = [os.path.join(orignal_image_folder, record) for record in os.listdir(
        orignal_image_folder) if os.path.isfile(os.path.join(orignal_image_folder, record))]

    n_tot = 1000
    n_epochs = 100
    batch_size = 50
    assert not n_tot % batch_size
    learning_rate = 0.0001

    iterator = DatasetIterator(filenames, n_epochs, batch_size, shuffle=True)
    colorizer = Colorizer(iterator, learning_rate)
    optimizer, loss_node = colorizer.training_op()

    saver = tf.train.Saver()

    load = False
    save = True

    count = 0
    epoch = 0
    losses = []
    with tf.Session() as sess:
        if load:
            saver.restore(sess, 'models/my-model')
        else:
            sess.run(tf.global_variables_initializer())
        try:
            while True:
                if not count % n_tot:
                    epoch += 1
                    count = 0
                    print("=== STARTED NEW EPOCH === number %d of %d" %
                          (epoch, n_epochs))
                    if save:
                        saver.save(sess, 'models/my-model')
                        print("model saved on epoch %d" % epoch)

                _, loss = sess.run(
                    [optimizer, loss_node])

                if not count % (1 * batch_size):
                    print("batch with count %d had loss: %f" % (count, loss))
                losses.append(loss)
                count += batch_size

        except tf.errors.OutOfRangeError:
            print("Done with training!")

        if save:
            saver.save(sess, 'models/my-model')
            print("model saved!")

            losses_filename = 'models/losses'
            np.save(losses_filename, losses)
            print("Losses saved in %s" % losses_filename)

        # plt.plot(losses)
        # plt.show()
