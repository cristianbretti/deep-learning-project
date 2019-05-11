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
from skimage import color

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


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _convert_to_example(filename, l_channel, labels, predicted_data):
    """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      l_channel: float ndarray, grayscale
      ab_channles: float ndarray, ab channels
      predicted_data: string, JPEG encoding of RGB predicted image (8, 8, 1536)
    Returns:
      Example proto
    """
    # predicted_data = np.reshape(predicted_data, (8*8*1536))

    example = tf.train.Example(features=tf.train.Features(feature={
        'filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'l_channel': _float_feature(l_channel.flatten()),
        'labels': _float_feature(labels.flatten()),
        'predicted': _float_feature(predicted_data.flatten()),
    }))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that resizes and grayscales data.
        self._small_image = tf.placeholder(dtype=tf.string)
        temp = tf.image.decode_jpeg(self._small_image, channels=3)
        temp = tf.reshape(temp, (1, 64, 64, 3))
        out_image = tf.image.resize_nearest_neighbor(temp, (128, 128))
        self._out_image = tf.reshape(out_image, (128, 128, 3))

        big = tf.image.resize_nearest_neighbor(temp, (299, 299))
        big_image = tf.reshape(big, (299, 299, 3))
        grayscaled_temp = tf.image.rgb_to_grayscale(big_image)
        stacked = tf.image.grayscale_to_rgb(grayscaled_temp)
        res = tf.cast(stacked, dtype=tf.float32)
        res = 2 * res / 255 - 1  # normalize between (-1, 1)
        self._gray_stacked = tf.reshape(res, (-1, 299, 299, 3))

    def resize_and_grayscale(self, image_data):
        image, stacked = self._sess.run([self._out_image, self._gray_stacked],
                                        feed_dict={self._small_image: image_data})
        return image, stacked


def _process_image(filename, coder, model):
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

    image, gray_stacked = coder.resize_and_grayscale(
        image_data)

    if model:
        predicted_data = model.predict(gray_stacked)
    else:
        predicted_data = np.zeros((3, 3))
        print("!!!!!!!!!!!NO MODEL!!!!!!!!!!")

    lab = color.rgb2lab(image).astype(np.float32)
    l_channel = 2 * lab[:, :, 0] / 100 - 1
    ab_channels = lab[:, :, 1:]

    ab_channels = ab_channels + 127  # set values between 0 and 254
    ab_channels = np.floor(ab_channels / 15)
    labels = ab_channels[:, :, 0] * 17 + ab_channels[:, :, 1]
    labels = labels.astype(int)

    return l_channel, labels, predicted_data


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

    if True:
        model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False, weights='imagenet', input_tensor=None, input_shape=(299, 299, 3), pooling=None)
        model.trainable = False
        print("Load model done")
        print("time taken")
        print(datetime.now() - zero_time)
    else:
        model = None

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d.tfrecord' % (
            name, shard, num_shards)
        output_file = os.path.join(output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(
            shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            orig_filename = orig_filenames[i]

            l_channel, labels, predicted_data = _process_image(
                orig_filename, coder, model)

            example = _convert_to_example(
                orig_filename, l_channel, labels, predicted_data)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 500:
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


def main(output_directory, num_shards):
    with open('datasets/tiny-imagenet-200/wnids.txt', 'r') as f:
        folder_names = f.readlines()
        orig_img_paths = []
        start = 0
        how_many = 20
        for i in range(start, start + how_many):
            orignal_image_folder = 'datasets/tiny-imagenet-200/train/' + \
                folder_names[i].strip() + '/images'
            orig_img_paths += [os.path.join(orignal_image_folder, im) for im in os.listdir(
                orignal_image_folder) if os.path.isfile(os.path.join(orignal_image_folder, im))]

        _process_image_files("train", orig_img_paths,
                             num_shards, output_directory)

        print("Started at %d and did %d folders, last one was %s" %
              (start, how_many, orig_img_paths[-1]))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage image_to_tfrecord <num partitions (multiples of 4)>")
    else:
        zero_time = datetime.now()
        main('datasets/preprocessed', int(sys.argv[1]))
