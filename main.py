import tensorflow as tf
import cv2
import numpy as np


inception_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
    include_top=False, weights='imagenet', input_tensor=None, input_shape=(299, 299, 3), pooling=None)
inception_model.trainable = False

# model = tf.keras.models.Sequential()
# model.add(inception_model)
# model.add(tf.keras.layers.Flatten())
# model.add(tf.layers.Dense(1000))

# fish = cv2.imread('fish.JPEG', cv2.IMREAD_COLOR)
# fish = cv2.resize(fish, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
# fish = fish.reshape(1, 299, 299, 3)
# x = model.predict(fish)
# print(np.argmax(x))
# print(x.shape)


grayscaled = np.empty((0, 299, 299, 1))
new_inputs = np.empty((0, 8, 8, 1536))

for image_nr in range(100):
    fish = cv2.imread(
        'datasets/tiny-imagenet-200/test/images/test_%d.JPEG' % image_nr, cv2.IMREAD_COLOR)
    fish = cv2.resize(fish, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(fish, cv2.COLOR_BGR2GRAY)
    gray = gray.reshape(1, 299, 299, 1)
    grayscaled = np.append(grayscaled, gray, axis=0)

    fish = fish.reshape(1, 299, 299, 3)
    predicted = inception_model.predict(fish)
    new_inputs = np.append(new_inputs, predicted, axis=0)


np.save('datasets/preprocessed/new', new_inputs)
np.save('datasets/preprocessed/grayscale', grayscaled)
