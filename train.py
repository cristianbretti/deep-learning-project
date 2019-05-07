import numpy as np
import os
import tensorflow as tf
from colorizer import Colorizer, DatasetIterator

if __name__ == "__main__":
    orignal_image_folder = 'datasets/preprocessed'

    filenames = [os.path.join(orignal_image_folder, record) for record in os.listdir(
        orignal_image_folder) if os.path.isfile(os.path.join(orignal_image_folder, record))]

    n_epochs = 1
    batch_size = 100

    iterator = DatasetIterator(filenames, n_epochs, batch_size)
    colorizer = Colorizer(iterator)
    optimizer, loss_node = colorizer.training_op()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                _, loss = sess.run([optimizer, loss_node])
                print(loss)
                saver.save(sess, 'models/my-model')
        except tf.errors.OutOfRangeError:
            pass
