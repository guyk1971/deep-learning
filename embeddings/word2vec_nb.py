import time

import numpy as np
import tensorflow as tf

import utils

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile


dataset_folder_path = 'data'
dataset_filename = 'text8.zip'
dataset_name = 'Text8 Dataset'


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


if not isfile(dataset_filename):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc=dataset_name) as pbar:
        urlretrieve(
            'http://mattmahoney.net/dc/text8.zip',
            dataset_filename,
            pbar.hook)

if not isdir(dataset_folder_path):
    with zipfile.ZipFile(dataset_filename) as zip_ref:
        zip_ref.extractall(dataset_folder_path)

with open('data/text8') as f:
    text = f.read()

################################################3
# preprocessing
words = utils.preprocess(text)
print(words[:30])

print("Total words: {}".format(len(words)))
print("Unique words: {}".format(len(set(words))))

vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]


#############################################################
# subsampling
## Your code here
th=1e-5
# my implementation - very inefficient !!
# def freq(word,corpus):
#     return corpus.count(word)/len(corpus)
#
# def prob(word,freqs,th):
#     return 1-np.sqrt(th/freqs[word])
#
# freqs = {word:freq(word,int_words) for word in int_words}
# p_drop = {word: prob(word,freqs,th) for word in int_words}
# train_words = {w for w in int_words if p_drop[w]>np.random.rand()}

# reference implementation
from collections import Counter
import random

word_counts=Counter(int_words) # dictionary like with k:v=int_words:count
total_count = len(int_words)
freqs={word: count/total_count for word,count in word_counts.items()}
p_drop={word: 1-np.sqrt(th/freqs[word]) for word in word_counts}
train_words = [word for word in int_words if p_drop[word]<random.random()]

##############################################################
# Making Batches
def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''

    # Your code here
    # get random number in the range (1,window_size) - this will be the number of words we'll take
    R=random.randint(1,window_size)
    # what about warping arond ? do we want to allow it ?
    start = max(idx-R,0)
    stop = min(idx+R+1,len(words))
    return words[start:idx]+words[idx+1:stop]

# note that the reference solution used np.random.randint
# note that the reference solution returned list(set(words[start:idx]+words[idx+1:stop])). not clear why the set() is needed...



def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''

    n_batches = len(words) // batch_size

    # only full batches
    words = words[:n_batches * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx + batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x] * len(batch_y))
        yield x, y


# note that we created a generator (because of the yield) . probably a recommended pattern for getting batches
# or any generator that has internal stage.


###############################################################################
# Building the graph

train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(dtype=tf.int32,shape=[None], name='inputs')
    labels = tf.placeholder(dtype=tf.int32,shape=[None,None],name='labels')       # ??? To make things work later, you'll need to set the second dimension of labels to None or 1.



# Embedding
n_vocab = len(int_to_vocab)
n_embedding =  200
with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_vocab,n_embedding),-1,1))
    embed = tf.nn.embedding_lookup(embedding,inputs)



################################################################################
# Negative sampling
# Number of negative labels to sample
n_sampled = 100
with train_graph.as_default():
    softmax_w =  tf.Variable(tf.truncated_normal((n_vocab,n_embedding),stddev=0.1))
    softmax_b =  tf.Variable(tf.zeros(n_vocab))


    # Calculate the loss using negative sampling
    loss = tf.nn.sampled_softmax_loss(softmax_w,softmax_b,labels,embed,n_sampled,n_vocab)

    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

with train_graph.as_default():
    ## From Thushan Ganegedara's implementation
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100
    # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent
    valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000, 1000 + valid_window), valid_size // 2))

    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

# If the checkpoints directory doesn't exist:
!mkdir checkpoints


################################################################################################
# Training
epochs = 10
batch_size = 1000
window_size = 10

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()
        for x, y in batches:

            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / 100),
                      "{:.4f} sec/batch".format((end - start) / 100))
                loss = 0
                start = time.time()

            if iteration % 1000 == 0:
                ## From Thushan Ganegedara's implementation
                # note that this is expensive (~20% slowdown if computed every 500 steps)
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)

            iteration += 1
    save_path = saver.save(sess, "checkpoints/text8.ckpt")
    embed_mat = sess.run(normalized_embedding)

###############################################################################################
# save checkpoint to be restored later
with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    embed_mat = sess.run(embedding)


#################################################################################################
# Visualization

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


viz_words = 500
tsne = TSNE()
embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])


fig, ax = plt.subplots(figsize=(14, 14))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)