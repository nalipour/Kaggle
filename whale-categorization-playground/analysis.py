import os.path

import pathlib
import imageio
import numpy as np
import pandas as pd

from skimage.color import rgb2gray
import tensorflow as tf
import matplotlib.pyplot as plt

training_labels = pd.read_csv('data/train.csv')
training_labels.sort_values('Image')
training_paths = pathlib.Path('data/train/').glob('*.jpg')
training_sorted = sorted([x for x in training_paths])
print('Found', len(training_sorted), 'training images')
print(*training_sorted[0:2], sep='\n')

# Plot data


def show_images(ims, cmap=None, labels=None):
    plt.figure(figsize=(3 * len(ims), 10))
    if labels is not None:
        assert(len(labels) == len(ims)), 'provide exactly one label per image'
    for idx, im in enumerate(ims):
        plt.subplot(1, len(ims), idx + 1)
        plt.imshow(im, cmap=cmap)
        # plt.axis('off')
        if labels is None:
            plt.title('Image ' + str(idx))
        else:
            plt.title(labels[idx])

    plt.tight_layout()
    plt.show()


ims = list(map(lambda p: imageio.imread(str(p)), training_sorted[0:4]))
show_images(ims)

# Gray scale

im = imageio.imread(str(training_sorted[0]))  # image instance for testing
im2 = imageio.imread(str(training_sorted[2]))  # image instance for testing

# Print the image dimensions
print('Original image shape: {}'.format(im.shape))

# Coerce the image into grayscale format (if not already)
print('New image shape: {}'.format(rgb2gray(im).shape))

MAX_INSTANCES = 4
ims = list(map(lambda p: rgb2gray(imageio.imread(str(p))), training_sorted[0:MAX_INSTANCES]))
show_images(ims, cmap='gray')

str(training_sorted[0])

X_train = [tf.Variable(im) for im in ims]
print(X_train[0].get_shape().as_list())
type(X_train[0])

training_labels['Id_int'] = pd.factorize(training_labels['Id'])[0]
training_labels = training_labels[0:MAX_INSTANCES]
num_classes = training_labels['Id_int'].nunique()
y_train = tf.one_hot(indices=training_labels['Id_int'],
                     depth=num_classes+1)

height = max([x.get_shape().as_list()[0] for x in X_train])
width = max([x.get_shape().as_list()[1] for x in X_train])

n_inputs = height * width
n_hidden1 = 300
n_hidden2 = 100
n_outputs = num_classes

X = tf.placeholder(tf.float32, shape=(None, height, width, 1), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, n_hidden1, name='hidden1',
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2',
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name='outputs')

with tf.name_scope('loss'):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                       logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
n_epochs = 2
batch_size = 1
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(MAX_INSTANCES//batch_size):
            X_batch, y_batch = tf.train.batch([X_train, y_train],
                                              batch_size=batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print(epoch, 'Train accuracy: ', acc_train)

    # result = sess.run(X_train)
