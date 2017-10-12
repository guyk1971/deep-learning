from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm


#----------------------------------------------------------------------------------------------------
# download the vgg net parameters
vgg_dir = 'tensorflow_vgg/'
# Make sure vgg exists
if not isdir(vgg_dir):
    raise Exception("VGG directory doesn't exist!")

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(vgg_dir + "vgg16.npy"):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='VGG16 Parameters') as pbar:
        urlretrieve(
            'https://s3.amazonaws.com/content.udacity-data.com/nd101/vgg16.npy',
            vgg_dir + 'vgg16.npy',
            pbar.hook)
else:
    print("Parameter file already exists!")

#----------------------------------------------------------------------------------------------------
# download the flowers dataset
import tarfile

dataset_folder_path = 'flower_photos'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile('flower_photos.tar.gz'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Flowers Dataset') as pbar:
        urlretrieve(
            'http://download.tensorflow.org/example_images/flower_photos.tgz',
            'flower_photos.tar.gz',
            pbar.hook)

if not isdir(dataset_folder_path):
    with tarfile.open('flower_photos.tar.gz') as tar:
        tar.extractall()
        tar.close()


#----------------------------------------------------------------------------------------------------
# Record the convnet code
import os

import numpy as np
import tensorflow as tf

from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

data_dir = 'flower_photos/'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]

# Set the batch size higher if you can fit in in your GPU memory
batch_size = 25
codes_list = []
labels = []
batch = []

codes = None

with tf.Session() as sess:
    vgg = vgg16.Vgg16()
    input_=tf.placeholder(tf.float32,[None,224,224,3])
    with tf.name_scope("content_vgg"):
        vgg.build(input_)
    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        files = os.listdir(class_path)
        for ii, file in enumerate(files, 1):
            # Add images to the current batch
            # utils.load_image crops the input images for us, from the center
            img = utils.load_image(os.path.join(class_path, file))
            batch.append(img.reshape((1, 224, 224, 3)))
            labels.append(each)

            # Running the batch through the network to get the codes
            if ii % batch_size == 0 or ii == len(files):

                # Image batch to pass to VGG network
                images = np.concatenate(batch)

                # TODO: Get the values from the relu6 layer of the VGG network
                codes_batch = sess.run(vgg.relu6,{input_:images})

                # Here I'm building an array of the codes
                if codes is None:
                    codes = codes_batch
                else:
                    codes = np.concatenate((codes, codes_batch))

                # Reset to start building the next batch
                batch = []
                print('{} images processed'.format(ii))

# write codes to file
with open('codes', 'w') as f:
    codes.tofile(f)

# write labels to file
import csv

with open('labels', 'w') as f:
    writer = csv.writer(f, delimiter='\n')
    writer.writerow(labels)

#----------------------------------------------------------------------------------------------------
# Building the classifier
# read codes and labels from file
import csv

with open('labels') as f:
    reader = csv.reader(f, delimiter='\n')
    labels = np.array([each for each in reader if len(each) > 0]).squeeze()
with open('codes') as f:
    codes = np.fromfile(f, dtype=np.float32)
    codes = codes.reshape((len(labels), -1))

#----------------------------------------------------------------------------------------------------
# Data prep
from sklearn import preprocessing

lb=preprocessing.LabelBinarizer()
lb.fit(labels)
labels_vecs = lb.transform(labels)


# split and shuffle the dataset using sklearn
from sklearn.model_selection import StratifiedShuffleSplit
sss=StratifiedShuffleSplit(1,test_size=0.2)
splitter = sss.split(codes,labels_vecs)
# get the indices for the taining and validation
train_idx, val_idx = next(splitter)
# now take 50% of the validation samples to be used as test set
val_set_size = int(len(val_idx)/2)
val_idx,test_idx=val_idx[:val_set_size],val_idx[val_set_size:]


train_x, train_y = codes[train_idx],labels_vecs[train_idx]
val_x, val_y = codes[val_idx], labels_vecs[val_idx]
test_x, test_y = codes[test_idx], labels_vecs[test_idx]


print("Train shapes (x, y):", train_x.shape, train_y.shape)
print("Validation shapes (x, y):", val_x.shape, val_y.shape)
print("Test shapes (x, y):", test_x.shape, test_y.shape)

#-------------------------------------------------------------------------------------
# Classifier layers


inputs_ = tf.placeholder(tf.float32, shape=[None, codes.shape[1]])
labels_ = tf.placeholder(tf.int64, shape=[None, labels_vecs.shape[1]])

# TODO: Classifier layers and operations
fc = tf.contrib.layers.fully_connected(inputs_,256)
logits = tf.layers.dense(fc,labels_vecs.shape[1],activation=None)

# cross entropy loss
ce = tf.nn.softmax_cross_entropy_with_logits(labels = labels_,logits = logits)
cost =  tf.reduce_mean(ce)
# training optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost)


# Operations for validation/test accuracy
predicted = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#-------------------------------------------------------------------------------------
# Batches!
def get_batches(x, y, n_batches=10):
    """ Return a generator that yields batches from arrays x and y. """
    batch_size = len(x) // n_batches

    for ii in range(0, n_batches * batch_size, batch_size):
        # If we're not on the last batch, grab data with size batch_size
        if ii != (n_batches - 1) * batch_size:
            X, Y = x[ii: ii + batch_size], y[ii: ii + batch_size]
            # On the last batch, grab the rest of the data
        else:
            X, Y = x[ii:], y[ii:]
        # I love generators
        yield X, Y


# -------------------------------------------------------------------------------------
# Training
num_epochs=100
saver = tf.train.Saver()
with tf.Session() as sess:
    # TODO: Your training code here
    sess.run(tf.global_variables_initializer())
    for epoch_i in range(num_epochs):

        for batch_i,(x,y) in enumerate(get_batches(train_x,train_y)):
            feed = {inputs_: x,
                    labels_: y}
            _,train_loss = sess.run([optimizer,cost],feed_dict=feed)
            print("Epoch:{}/{}".format(epoch_i,num_epochs),"Iteration:{}".format(batch_i),"train loss:{:.5f}".format(train_loss))

            if batch_i % 5 == 0: #every 5 iterations check accuracy on validation set
                feed = {inputs_: val_x, labels_: val_y}
                val_acc = sess.run(accuracy,feed_dict=feed)
                print("------Epoch:{}/{}".format(epoch_i, num_epochs), "Iteration:{}".format(batch_i),
                      "Validation acc:{:.4f}".format(val_acc))




    saver.save(sess, "checkpoints/flowers.ckpt")


# -------------------------------------------------------------------------------------
# Testing

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    feed = {inputs_: test_x,
            labels_: test_y}
    test_acc = sess.run(accuracy, feed_dict=feed)
    print("Test accuracy: {:.4f}".format(test_acc))

import matplotlib.pyplot as plt
from scipy.ndimage import imread

test_img_path = 'flower_photos/roses/10894627425_ec76bbc757_n.jpg'
test_img = imread(test_img_path)
plt.imshow(test_img)

with tf.Session() as sess:
    img = utils.load_image(test_img_path)
    img = img.reshape((1, 224, 224, 3))

    feed_dict = {input_: img}
    code = sess.run(vgg.relu6, feed_dict=feed_dict)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    feed = {inputs_: code}
    prediction = sess.run(predicted, feed_dict=feed).squeeze()