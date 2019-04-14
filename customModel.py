from __future__ import division, print_function, unicode_literals
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
import json
import collections
from PIL import Image
import glob
from time import time
import dataService


Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1")
height, width = hub.get_expected_image_size(module)
batch_size = 2151
validation_size = 400

class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 3)
      assert images.shape[3] == 3
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2] * images.shape[3])
      if dtype == tf.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def generate_datasets():

    class DataSets(object):
        pass
    data_sets = DataSets()

    # Loading the data.
    data = dataService.getData()
    images = np.zeros((batch_size, height, width, 3)) ## fill with your data
    labels = np.zeros((batch_size))
    x = 0
    for car in data:
        try:
            
            img = Image.open(BytesIO(base64.b64decode(car['image'])))
            img = img.resize((width, height), Image.NEAREST)      # use nearest neighbour
            #parameters = car[]
            images[x] = img
            labels[x] = car['Parduota per (pagal taisykles)']
            x += 1
        except Exception as identifier:
            print(identifier)
            pass


    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    test_images = images[1:100]
    test_labels = labels[1:100]
 
    data_sets.train = DataSet(train_images, train_labels, dtype=tf.float32)
    data_sets.validation = DataSet(validation_images, validation_labels, dtype=tf.float32)
    data_sets.test = DataSet(test_images, test_labels, dtype=tf.float32)


    return data_sets




def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


car_dataset = generate_datasets()

images = np.zeros((batch_size, height, width, 3)) ## fill with your data
filenames = glob.glob("C:\\Users\\Ignas\\Documents\\Kursinis\\allcars\\*.jpg")


for x in range(0, batch_size):
    img = Image.open(filenames[x])
    img = img.resize((width, height), Image.NEAREST)      # use nearest neighbour
    images[x] = img


features = module(images)  # Features(tf graph) with shape [batch_size, num_features].

## THE FNN PART

# correct labels
y_ = tf.placeholder(tf.float32, [None, 30])

print(tf.shape(features))
print(features.get_shape())
# build the network
keep_prob_input = tf.placeholder(tf.float32)
x_drop = tf.nn.dropout(features, keep_prob=keep_prob_input)

W_fc1 = weight_variable([2048, 1200])
b_fc1 = bias_variable([1200])
h_fc1 = tf.nn.relu(tf.matmul(x_drop, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1200, 1200])
b_fc2 = bias_variable([1200])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([1200, 30])
b_fc3 = bias_variable([30])
y = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

# define the loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# define training step and accuracy
train_step = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create a saver
# saver = tf.train.Saver()

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    f = sess.run([features]) # Features numpy to save it 

    # train
    batch_size = 100
    print("Startin Burn-In...")
    saver.save(sess, 'car_dataset')
    for i in range(700):
        input_images, correct_predictions = car_dataset.train.next_batch(batch_size)
        if i % (60000/batch_size) == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: input_images, y_: correct_predictions, keep_prob_input: 1.0, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            # validate
            test_accuracy = sess.run(accuracy, feed_dict={
                x: car_dataset.test.images, y_: car_dataset.test.labels, keep_prob_input: 1.0, keep_prob: 1.0})
            print("Validation accuracy: %g." % test_accuracy)
        sess.run(train_step, feed_dict={x: input_images, y_: correct_predictions, keep_prob_input: 0.8, keep_prob: 0.5})
    saver.restore(sess, 'car_dataset_fc_best')
    print("Starting the training...")
    start_time = time()
    best_accuracy = 0.0
    for i in range(20*60000/batch_size):
        input_images, correct_predictions = car_dataset.train.next_batch(batch_size)
        if i % (60000/batch_size) == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: input_images, y_: correct_predictions, keep_prob_input: 1.0, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            # validate
            test_accuracy = sess.run(accuracy, feed_dict={
                x: car_dataset.test.images, y_: car_dataset.test.labels, keep_prob_input: 1.0, keep_prob: 1.0})
            if test_accuracy >= best_accuracy:
                saver.save(sess, 'car_dataset_fc_best')
                best_accuracy = test_accuracy
                print("Validation accuracy improved: %g. Saving the network." % test_accuracy)
            else:
                saver.restore(sess, 'car_dataset_fc_best')
                print("Validation accuracy was: %g. It was better before: %g. " % (test_accuracy, best_accuracy) +
                      "Using the old params for further optimizations.")
        sess.run(train_step, feed_dict={x: input_images, y_: correct_predictions, keep_prob_input: 0.8, keep_prob: 0.5})
    print("The training took %.4f seconds." % (time() - start_time))

    # validate
    print("Best test accuracy: %g" % best_accuracy)