# A python file to support tv script generation
import helper
#------------------------------------------------------------
data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]

#------------------------------------------------------------
# Explore the data

view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
#-----------------------------------------------------------------------------------------------------------------------
# Implement Preprocessing Functions
#------------------------------------------------------------
# Lookup table

import numpy as np
import problem_unittests as tests
from collections import Counter

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_counts = Counter(text) # creates a dictionary word:count
    sorted_vocab = sorted(word_counts,key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii,word in enumerate(sorted_vocab)}
    vocab_to_int = {word:ii for ii,word in enumerate(sorted_vocab)}
    return (vocab_to_int, int_to_vocab)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)

#------------------------------------------------------------
# Tokenize Punctuation

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    Tokenize=dict()
    Tokenize['.']='<PERIOD>'
    Tokenize[','] = '<COMMA>'
    Tokenize['"'] = '<QUOTATION_MARK>'
    Tokenize[';'] = '<SEMICOLON>'
    Tokenize['!'] = '<EXCLAMATION_MARK>'
    Tokenize['?'] = '<QUESTION_MARK>'
    Tokenize['('] = '<LEFT_PAREN>'
    Tokenize[')'] = '<RIGHT_PAREN>'
    Tokenize['--'] = '<DASH>'
    Tokenize['\n'] = '<RETURN>'
    return Tokenize

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)

#------------------------------------------------------------
# Preprocess data and save it
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

#-----------------------------------------------------------------------------------------------------------------------
# Check point
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests
import tensorflow as tf

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

#-----------------------------------------------------------------------------------------------------------------------
# Build the neural network

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

#------------------------------------------------------------
# Input

# Implement the get_inputs() function to create TF Placeholders for the Neural Network. It should create the following placeholders:
# Input text placeholder named "input" using the TF Placeholder name parameter.
# Targets placeholder
# Learning Rate placeholder
# Return the placeholders in the following tuple (Input, Targets, LearningRate)

def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    inputs = tf.placeholder(dtype=tf.int32, shape=(None,None), name='input')
    targets = tf.placeholder(dtype=tf.int32, shape=(None,None), name='targets')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    # theoretically, keep_prob placeholder for dropout should have been added here as well
    # and also returned as one of the inputs (so that testing and training will use different values as needed
    # But its omitted to pass the unit testing script

    return (inputs,targets,learning_rate)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)

#------------------------------------------------------------
# Build RNN Cell and Initialize

# The Rnn size should be set using rnn_size
# Initalize Cell State using the MultiRNNCell's zero_state() function
# Apply the name "initial_state" to the initial state using tf.identity()
# Return the cell and initial state in the following tuple (Cell, InitialState)

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    num_layers = 3
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    # todo: consider dropout
    # keep_prob = 0.5
    # drop = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([lstm]*num_layers)    # if dropout applied then replace 'lstm' with 'drop'
    initial_state=tf.identity(cell.zero_state(batch_size,tf.float32),name='initial_state')
    return (cell,initial_state)



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)

#------------------------------------------------------------
# word embedding
# apply word embedding using tensorflow. return the embedded sequence

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size,embed_dim),-1,1))
    embed = tf.nn.embedding_lookup(embedding,input_data)
    return embed


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)

#------------------------------------------------------------
# build RNN

# You created a RNN Cell in the get_init_cell() function. Time to use the cell to create a RNN.
# Build the RNN using the tf.nn.dynamic_rnn()
# Apply the name "final_state" to the final state using tf.identity()
# Return the outputs and final_state state in the following tuple (Outputs, FinalState)

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs,state = tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32)     # why dtype=tf.float32 ? doesnt work with int32
    final_state = tf.identity(state,name='final_state')

    return (outputs,final_state)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)

#------------------------------------------------------------
# Build the neural network
# Apply the functions you implemented above to:
#   Apply embedding to input_data using your get_embed(input_data, vocab_size, embed_dim) function.
#   Build RNN using cell and your build_rnn(cell, inputs) function.
#   Apply a fully connected layer with a linear activation and vocab_size as the number of outputs.
# Return the logits and final state in the following tuple (Logits, FinalState)
def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """

    embed_words = get_embed(input_data, vocab_size, embed_dim) # shape : [None, None, 300]
    rnn_outputs, final_state = build_rnn(cell, embed_words) # shape: [None, None, 256]
    logits = tf.layers.dense(rnn_outputs,vocab_size)
    return (logits,final_state)



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)

#------------------------------------------------------------
# Batches

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    num_batches = len(int_text)//(batch_size*seq_length)
    trimmed_text = int_text[:num_batches*(batch_size*seq_length)]
    inputs = trimmed_text
    targets = trimmed_text[1:]+[trimmed_text[0]]
    # the below code was inspired (copied with modification) from KaRNNa exercise - get_batches)
    inputs = np.reshape(inputs,(batch_size,-1))     # now inputs.shape=[batch_size,num_batches*seq_length]
    targets = np.reshape(targets, (batch_size, -1))
    Batches=np.zeros((num_batches,2,batch_size,seq_length))
    for b,n in enumerate(range(0,inputs.shape[1],seq_length)):
        inp=np.expand_dims(inputs[:,n:n+seq_length],0)
        tar=np.expand_dims(targets[:,n:n+seq_length],0)
        Batches[b]=np.vstack((inp,tar))
    return Batches


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_batches(get_batches)

#-----------------------------------------------------------------------------------------------------------------------
# Neural Network Training

# RNN Size
rnn_size = 256
# Embedding Dimension Size
embed_dim = 200
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'

# Build the graph
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

#------------------------------------------------------------
# Train
# Number of Epochs
num_epochs = 100 #20
# Batch Size
batch_size = 23 #10 (1 epoch is 230 batches)
# Sequence Length
seq_length = 30 #30
# Learning Rate
learning_rate = 0.005
# Show stats for every n number of batches
show_every_n_batches = 20


batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
#-----------------------------------------------------------------------------------------------------------------------
# Save Parameters
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))
#-----------------------------------------------------------------------------------------------------------------------
# Check point
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()

#-----------------------------------------------------------------------------------------------------------------------
# Implement Generate Functions
# Get Tensors
# Get tensors from loaded_graph using the function get_tensor_by_name(). Get the tensors using the following names:
# "input:0"
# "initial_state:0"
# "final_state:0"
# "probs:0"
# Return the tensors in the following tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)

def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    InputTensor = loaded_graph.get_tensor_by_name("input:0")
    InitialStateTensor = loaded_graph.get_tensor_by_name("initial_state:0")
    FinalStateTensor = loaded_graph.get_tensor_by_name("final_state:0")
    ProbsTensor = loaded_graph.get_tensor_by_name("probs:0")
    return InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_tensors(get_tensors)


#------------------------------------------------------------
# Choose word
def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """

    sampled_int=np.random.choice(len(int_to_vocab),1,p=probabilities)[0]
    return int_to_vocab[sampled_int]


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_pick_word(pick_word)

#------------------------------------------------------------
# Generate TV Script
gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})

        pred_word = pick_word(probabilities[dyn_seq_length - 1], int_to_vocab)

        gen_sentences.append(pred_word)

    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')

    print(tv_script)