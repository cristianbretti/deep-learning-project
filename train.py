import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from colorizer import Colorizer, DatasetIterator

if __name__ == "__main__":
    orignal_image_folder = 'datasets/preprocessed_1000'

    filenames = [os.path.join(orignal_image_folder, record) for record in os.listdir(
        orignal_image_folder) if os.path.isfile(os.path.join(orignal_image_folder, record))]

    n_tot = 1000
    n_epochs = 2
    batch_size = 100
    learning_rate = 0.001

    iterator = DatasetIterator(filenames, n_epochs, batch_size)
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
                    print("=== STARTED NEW EPOCH === number %d of %d" %
                          (epoch, n_epochs))
                    if save:
                        saver.save(sess, 'models/my-model')
                        print("model saved on epoch %d" % epoch)
                    epoch += 1
                    count = 0

                _, loss = sess.run(
                    [optimizer, loss_node])

                if not count % (2 * batch_size):
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

        plt.plot(losses)
        plt.show()
