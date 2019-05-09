import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from colorizer import Colorizer, DatasetIterator
import time


if __name__ == "__main__":
    orignal_image_folder = 'datasets/preprocessed'

    filenames = [os.path.join(orignal_image_folder, record) for record in os.listdir(
        orignal_image_folder) if os.path.isfile(os.path.join(orignal_image_folder, record))]

    n_tot = 50
    n_epochs = 50
    batch_size = 1
    # assert not batch_size % n_tot
    learning_rate = 0.0001

    iterator = DatasetIterator(filenames, n_epochs, batch_size, shuffle=True)
    colorizer = Colorizer(iterator, learning_rate)
    # predicted_node, one_hot_node, labels_node, one_hot_data_node = colorizer.prepare_next_data_batch()
    optimizer, loss_node = colorizer.training_op()

    saver = tf.train.Saver()

    load = False
    save = True

    count = 0
    epoch = 0
    losses = []

    total_bin_ab_time = 0
    total_train_time = 0

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

                # predicted, one_hot, labels, one_hot_data = sess.run(
                #     [predicted_node, one_hot_node, labels_node, one_hot_data_node])

                # labels = colorizer.ab_to_bin(ab_channels_real)

                _, loss = sess.run(
                    [optimizer, loss_node])

                # ,
                # feed_dict={
                #     colorizer.predicted_input: predicted,
                #     colorizer.labels: labels
                # })

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

        plt.plot(losses)
        plt.show()
