# coding: utf-8
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts image data to TFRecords file format with Example protos.

The image data set is expected to reside in JPEG files located in the
following directory structure.

  data_dir/image0.jpeg
  data_dir/image1.jpg
  ...

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# TODO: uncomment!
# inception_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
#     include_top=False, weights='imagenet', input_tensor=None, input_shape=(299, 299, 3), pooling=None)
# inception_model.trainable = False

tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads to preprocess the images.')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(orig_filename, orig_image_buffer, orig_height, orig_width):
    """Build an Example proto for an example.

    Args:
      orig_filename: string, path to an image file, e.g., '/path/to/example.JPG'
      orig_image_buffer: string, JPEG encoding of RGB image
      orig_height: integer, image height in pixels
      orig_width: integer, image width in pixels
    Returns:
      Example proto
    """

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'orig/image/height': _int64_feature(orig_height),
        'orig/image/width': _int64_feature(orig_width),
        'orig/image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'orig/image/channels': _int64_feature(channels),
        'orig/image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        'orig/image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(orig_filename))),
        'orig/image/encoded': _bytes_feature(tf.compat.as_bytes(orig_image_buffer))
    }))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(
            image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _is_png(filename):
    """Determine if a file contains a PNG format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a PNG.
    """
    return '.png' in filename


def _process_image(filename, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # TODO: make code have a function that does all this, should return
    # a encoded image! not decoded!

    # image = cv2.resize(image, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # image = image.reshape(1, 299, 299, 3)
    # predicted = inception_model.predict(image)
    # predicted = predicted.reshape(8, 8, 1536)
    # image = image.reshape(299, 299, 3)

    # return image_data, height, width
    return image_data, 64, 64


def _process_image_files_batch(coder, thread_index, ranges, name, orig_filenames, num_shards, output_directory):
    """Processes and saves list of images as TFRecord in 1 thread.

    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      orig_filenames: list of strings; each string is a path to an image file
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(
            shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            orig_filename = orig_filenames[i]

            orig_image_buffer, orig_height, orig_width = _process_image(
                orig_filename, coder)

            example = _convert_to_example(
                orig_filename, orig_image_buffer, orig_height, orig_width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, orig_filenames, num_shards, output_directory):
    """Process and save list of images as TFRecord of Example protos.

    Args:
      name: string, unique identifier specifying the data set
      orig_filenames: list of strings; each string is a path to an image file
      num_shards: integer number of shards for this data set.
      output_directory : Directory for output files
    """
    spacing = np.linspace(0, len(orig_filenames),
                          FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' %
          (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name,
                orig_filenames, num_shards, output_directory)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(orig_filenames)))
    sys.stdout.flush()


def main(orignal_image_folder, output_directory, num_shards):

    orig_img_paths = [os.path.join(orignal_image_folder, im) for im in os.listdir(
        orignal_image_folder) if os.path.isfile(os.path.join(orignal_image_folder, im))]

    _process_image_files("cool", orig_img_paths, num_shards, output_directory)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage image_to_tfrecord <input_images_folder> <num partitions (multiples of 4)>")
    else:
        main(sys.argv[1], 'test-output', int(sys.argv[2]))


# For reading files

# filename = "test-output/cool-00000-of-00004"
# sess = tf.Session()

# for serialized_example in tf.python_io.tf_record_iterator(filename):
#     example = tf.train.Example()
#     example.ParseFromString(serialized_example)

#     # traverse the Example format to get data
#     img = example.features.feature['origimage/encoded']

#     # get the data out of tf record
#     orignal_image_height = example.features.feature['orig/image/height']
#     orignal_image_width = example.features.feature['orig/image/width']
#     orignal_image_colors = example.features.feature['orig/image/colorspace']
#     orignal_image_channels = example.features.feature['orig/image/channels']
#     orignal_image_format = example.features.feature['orig/image/format']
#     orignal_image_filename = example.features.feature['orig/image/filename']
#     orignal_image_data = example.features.feature['orig/image/encoded']

#     orignal_image = sess.run(tf.image.decode_jpeg(
#         orignal_image_data.bytes_list.value[0], channels=3))

#     plt.subplot(111)
#     plt.title("Image Name : " + str(orignal_image_filename.bytes_list.value[0]) + "\n" +
#               "Image Height : " + str(orignal_image_height.int64_list.value[0]) + "\n" +
#               "Image Weight : " + str(orignal_image_width.int64_list.value[0]) + "\n" +
#               "Image ColourSpace : " + str(orignal_image_colors.bytes_list.value[0]) + "\n" +
#               "Image Channels : " + str(orignal_image_channels.int64_list.value[0]) + "\n" +
#               "Image format : " + str(orignal_image_format.bytes_list.value[0]) + "\n")
#     plt.imshow(orignal_image)

#     plt.show()
#     break
