import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical

reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)

# counting word frequency
from collections import Counter
total_counts = Counter()
for idx,row in reviews.iterrows():
    total_counts.update(row[0].split(' '))

print("Total words in data set: ", len(total_counts))

# get the first 10K most frequent words
vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]
print(vocab[:60])
# show the last word and see if its frequent enough (helps to set the number of words we keep)
print(vocab[-1], ': ', total_counts[vocab[-1]])


# Create a dictionary called word2idx that maps each word in the vocabulary to an index.
# The first word in vocab has index 0, the second word has index 1, and so on.

word2idx = {v:k for k,v in enumerate(vocab)}

# text to vector function
def text_to_vector(text):
    word_vector = np.zeros(len(vocab))
    for w in text.split(' '):
        if w in vocab:
            word_vector[word2idx[w]] += 1

    return word_vector


text_to_vector('The tea is for a party to celebrate '
               'the movie so she has no time for a cake')[:65]


word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
for ii, (_, text) in enumerate(reviews.iterrows()):
    word_vectors[ii] = text_to_vector(text[0])

# Printing out the first 5 word vectors
word_vectors[:5, :23]


# Train, Validation and test sets
Y = (labels=='positive').astype(np.int_)
records = len(labels)

shuffle = np.arange(records)
np.random.shuffle(shuffle)
test_fraction = 0.9

train_split, test_split = shuffle[:int(records*test_fraction)], shuffle[int(records*test_fraction):]
trainX, trainY = word_vectors[train_split,:], to_categorical(Y.values[train_split], 2)
testX, testY = word_vectors[test_split,:], to_categorical(Y.values[test_split], 2)

# Building the network
def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()

    # Inputs
    net = tflearn.input_data([None, 10000])

    # Hidden layer(s)
    net = tflearn.fully_connected(net, 200, activation='ReLU')
    net = tflearn.fully_connected(net, 25, activation='ReLU')

    # Output layer
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd',
                             learning_rate=0.1,
                             loss='categorical_crossentropy')

    model = tflearn.DNN(net)
    return model


model=build_model()

# Training the network
# Training
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=100)
