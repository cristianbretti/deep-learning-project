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

        losses_val_filename = 'models/losses_val'
        np.save(losses_val_filename, losses_val)
        print("Losses_val saved in %s" % losses_val_filename)

    show_losses()
    exit(0)


def show_losses():
    measures_per_epoch = n_tot/batch_size

    plt.title('Training and validation loss graph')
    plt.plot(losses, 'r')
    x_val = np.arange(len(losses_val) + 1)[1:] * measures_per_epoch
    plt.plot(x_val, losses_val, 'g')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(np.arange(0, len(losses), step=measures_per_epoch),
               np.arange(n_epochs))
    plt.show()


signal.signal(signal.SIGINT, signal_handler)
if __name__ == "__main__":
    orignal_image_folder = 'datasets/preprocessed_cat'

    filenames = [os.path.join(orignal_image_folder, record) for record in os.listdir(
        orignal_image_folder) if os.path.isfile(os.path.join(orignal_image_folder, record))]

    test_image_folder = 'datasets/preprocessed_cat_val'
    val_filenames = [os.path.join(test_image_folder, record) for record in os.listdir(
        test_image_folder) if os.path.isfile(os.path.join(test_image_folder, record))]

    n_tot = 1000
    n_epochs = 20
    batch_size = 50
    assert not n_tot % batch_size
    val_batch_size = 50
    learning_rate = 0.0001

    iterator = DatasetIterator(filenames, batch_size=batch_size, shuffle=True)

    val_iterator = DatasetIterator(
        val_filenames, batch_size=val_batch_size, shuffle=False)

    colorizer = Colorizer(iterator, learning_rate)
    optimizer, loss_node = colorizer.training_op()

    colorizer.set_iterator(val_iterator)
    _, val_loss_node = colorizer.training_op()

    saver = tf.train.Saver()

    load = False
    save = True

    losses = []
    losses_val = []
    with tf.Session() as sess:
        if load:
            saver.restore(sess, 'models/my-model')
        else:
            sess.run(tf.global_variables_initializer())

        for epoch in range(1, n_epochs + 1):
            print("=== STARTED NEW EPOCH === number %d of %d" %
                  (epoch, n_epochs))
            sess.run(iterator.initializer())
            try:
                while True:

                    _, loss = sess.run(
                        [optimizer, loss_node])

                    print("...loss...: %f" % (loss))
                    losses.append(loss)

            except tf.errors.OutOfRangeError:
                print("Done epoch %d training!" % epoch)
                if save:
                    saver.save(sess, 'models/my-model')
                    print("model saved on epoch %d" % epoch)

                print("Calculation validation loss")
                sess.run(val_iterator.initializer())
                total_loss = 0
                val_count = 0
                try:
                    while True:
                        val_loss = sess.run(val_loss_node)
                        val_count += 1
                        total_loss += val_loss
                except tf.errors.OutOfRangeError:
                    total_loss = total_loss / val_count
                    losses_val.append(total_loss)
                    print("Validation loss for epoch %d is: %f" %
                          (epoch, total_loss))
                    colorizer.set_iterator(iterator)

        print("DONE WITH TRAINING!")
        if save:
            saver.save(sess, 'models/my-model')
            print("model saved!")

            losses_filename = 'models/losses'
            np.save(losses_filename, losses)
            print("Losses saved in %s" % losses_filename)

            losses_val_filename = 'models/losses_val'
            np.save(losses_val_filename, losses_val)
            print("Losses_val saved in %s" % losses_val_filename)

        show_losses()
