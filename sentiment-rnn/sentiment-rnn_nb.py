import numpy as np
import tensorflow as tf

with open('../sentiment-network/reviews.txt', 'r') as f:
    reviews = f.read()
with open('../sentiment-network/labels.txt', 'r') as f:
    labels = f.read()



#-----------------------------------------------------------------------------------------------
# Data preprocessing
from string import punctuation
all_text = ''.join([c for c in reviews if c not in punctuation])
reviews = all_text.split('\n')

all_text = ' '.join(reviews)
words = all_text.split()

# Encoding the words
# Create your dictionary that maps vocab words to integers here
from collections import Counter
word_counts = Counter(words)  # create a sort of dictionary k,v=word,count
sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)  # list of words sorted by their word count
vocab_to_int = {word: ii+1 for ii, word in enumerate(sorted_vocab)}

# Convert the reviews to integers, same shape as reviews list, but with integers
reviews_ints = [[vocab_to_int[w] for w in review.split()] for review in reviews]

labels = [int(w=='positive') for w in labels.split()]


from collections import Counter
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

# Filter out that review with 0 length
reviews_ints.remove([])


seq_len = 200
features = np.array([[0]*(seq_len-len(r))+r[:seq_len] for r in reviews_ints])

split_frac = 0.8
split_idx=int(split_frac*features.shape[0])
train_x, val_x = features[:split_idx,:], features[split_idx:,:]
train_y, val_y = np.array(labels[:split_idx]), np.array(labels[split_idx:])

val_idx = int(0.5 * val_x.shape[0])
val_x, test_x = val_x[:val_idx,:], val_x[val_idx:,:]
val_y, test_y = val_y[:val_idx], val_y[val_idx:]


print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

#=============================================================================================
# Build the graph

lstm_size = 256
lstm_layers = 1
batch_size = 500
learning_rate = 0.001


n_words = len(vocab_to_int)

# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(dtype=tf.int32,shape=[None,None], name='inputs')   # shape = [Batch_size,Seq_size] left as None
    labels_ = tf.placeholder(dtype=tf.int32,shape=[None,1],name='labels')    # shape = [Batch_size, label_size]
    keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')

# Embedding
embed_size = 300
with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words,embed_size),-1,1))
    embed = tf.nn.embedding_lookup(embedding,inputs_)


# LSTM Cell
with graph.as_default():
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=keep_prob)

    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop]*lstm_layers)

    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)


with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell,embed,initial_state=initial_state)

with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]



# Training
epochs = 10

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)

        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

            if iteration % 5 == 0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            if iteration % 25 == 0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration += 1
    saver.save(sess, "checkpoints/sentiment.ckpt")


test_acc = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))