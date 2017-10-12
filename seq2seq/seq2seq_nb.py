# Dataset
import helper

source_path = 'data/letters_source.txt'
target_path = 'data/letters_target.txt'

source_sentences = helper.load_data(source_path)
target_sentences = helper.load_data(target_path)

source_sentences[:50].split('\n')


# Preprocess
# turn the characters to a list of integers
def extract_character_vocab(data):
    special_words = ['<pad>', '<unk>', '<s>',  '<\s>']

    set_words = set([character for line in data.split('\n') for character in line])
    int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + list(set_words))}
    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int

# Build int2letter and letter2int dicts
# Question : why do we need distinct dictionaries for source and target ?
# Question: why eventually they end up being the same dictionary when the order of the letters is different ?
source_int_to_letter, source_letter_to_int = extract_character_vocab(source_sentences)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_sentences)

# Convert characters to ids
source_letter_ids = [[source_letter_to_int.get(letter, source_letter_to_int['<unk>']) for letter in line] for line in source_sentences.split('\n')]
target_letter_ids = [[target_letter_to_int.get(letter, target_letter_to_int['<unk>']) for letter in line] for line in target_sentences.split('\n')]

print("Example source sequence")
print(source_letter_ids[:3])
print("\n")
print("Example target sequence")
print(target_letter_ids[:3])

# find the longest sequence and pad all the rest to get to that length:
def pad_id_sequences(source_ids, source_letter_to_int, target_ids, target_letter_to_int, sequence_length):
    new_source_ids = [sentence + [source_letter_to_int['<pad>']] * (sequence_length - len(sentence)) \
                      for sentence in source_ids]
    new_target_ids = [sentence + [target_letter_to_int['<pad>']] * (sequence_length - len(sentence)) \
                      for sentence in target_ids]

    return new_source_ids, new_target_ids


# Use the longest sequence as sequence length
# get the max value of the concatenated list of sentences length 
sequence_length = max(
        [len(sentence) for sentence in source_letter_ids] + [len(sentence) for sentence in target_letter_ids])

# Pad all sequences up to sequence length
source_ids, target_ids = pad_id_sequences(source_letter_ids, source_letter_to_int, 
                                          target_letter_ids, target_letter_to_int, sequence_length)

print("Sequence Length")
print(sequence_length)
print("\n")
print("Input sequence example")
print(source_ids[:3])
print("\n")
print("Target sequence example")
print(target_ids[:3])


# Model
from distutils.version import LooseVersion
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))


# Hyper Parameters
# Number of Epochs
epochs = 60
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size - why is this size of the embedding ? 
encoding_embedding_size = 13
decoding_embedding_size = 13
# Learning Rate
learning_rate = 0.001

# Network Input
input_data = tf.placeholder(tf.int32, [batch_size, sequence_length])
targets = tf.placeholder(tf.int32, [batch_size, sequence_length])
lr = tf.placeholder(tf.float32)

# Encoding
source_vocab_size = len(source_letter_to_int)

# Encoder embedding
enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

# Encoder
enc_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)
_, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input, dtype=tf.float32)


# process Decoding input
import numpy as np

# Process the input we'll feed to the decoder
# we prune the last character in the sequence and concatenate <s> in the beginning. why ?
ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])        # what is the purpose in this line ?
dec_input = tf.concat([tf.fill([batch_size, 1], target_letter_to_int['<s>']), ending], 1)

demonstration_outputs = np.reshape(range(batch_size * sequence_length), (batch_size, sequence_length))

sess = tf.InteractiveSession()
print("Targets")
print(demonstration_outputs[:2])
print("\n")
print("Processed Decoding Input")
print(sess.run(dec_input, {targets: demonstration_outputs})[:2])


# Decoding
target_vocab_size = len(target_letter_to_int)

# Decoder Embedding
dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

# Decoder RNNs
dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)

with tf.variable_scope("decoding") as decoding_scope:
    # Output Layer
    output_fn = lambda x: tf.contrib.layers.fully_connected(x, target_vocab_size, None, scope=decoding_scope)



#Decoder During Training
with tf.variable_scope("decoding") as decoding_scope:
    # Training Decoder
    train_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(enc_state)
    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
        dec_cell, train_decoder_fn, dec_embed_input, sequence_length, scope=decoding_scope)

    # Apply output function
    train_logits = output_fn(train_pred)


# Decoder during inference
with tf.variable_scope("decoding", reuse=True) as decoding_scope:
    # Inference Decoder
    infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
        output_fn, enc_state, dec_embeddings, target_letter_to_int['<s>'], target_letter_to_int['<\s>'],
        sequence_length - 1, target_vocab_size)
    inference_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, infer_decoder_fn, scope=decoding_scope)


# optimization
# Our loss function is tf.contrib.seq2seq.sequence_loss provided by the tensor flow seq2seq module.
# It calculates a weighted cross-entropy loss for the output logits.

# Loss function
cost = tf.contrib.seq2seq.sequence_loss(
    train_logits,
    targets,
    tf.ones([batch_size, sequence_length]))

# Optimizer
optimizer = tf.train.AdamOptimizer(lr)

# Gradient Clipping
gradients = optimizer.compute_gradients(cost)
capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
train_op = optimizer.apply_gradients(capped_gradients)


# Train
import numpy as np

train_source = source_ids[batch_size:]
train_target = target_ids[batch_size:]

valid_source = source_ids[:batch_size]
valid_target = target_ids[:batch_size]

sess.run(tf.global_variables_initializer())

for epoch_i in range(epochs):
    for batch_i, (source_batch, target_batch) in enumerate(
            helper.batch_data(train_source, train_target, batch_size)):
        _, loss = sess.run(
            [train_op, cost],
            {input_data: source_batch, targets: target_batch, lr: learning_rate})
        batch_train_logits = sess.run(
            inference_logits,
            {input_data: source_batch})
        batch_valid_logits = sess.run(
            inference_logits,
            {input_data: valid_source})

        train_acc = np.mean(np.equal(target_batch, np.argmax(batch_train_logits, 2)))
        valid_acc = np.mean(np.equal(valid_target, np.argmax(batch_valid_logits, 2)))
        print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}'
              .format(epoch_i, batch_i, len(source_ids) // batch_size, train_acc, valid_acc, loss))



# Prediction
input_sentence = 'hello'


input_sentence = [source_letter_to_int.get(word, source_letter_to_int['<unk>']) for word in input_sentence.lower()]
input_sentence = input_sentence + [0] * (sequence_length - len(input_sentence))
batch_shell = np.zeros((batch_size, sequence_length))
batch_shell[0] = input_sentence
chatbot_logits = sess.run(inference_logits, {input_data: batch_shell})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in input_sentence]))
print('  Input Words: {}'.format([source_int_to_letter[i] for i in input_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in np.argmax(chatbot_logits, 1)]))
print('  Chatbot Answer Words: {}'.format([target_int_to_letter[i] for i in np.argmax(chatbot_logits, 1)]))