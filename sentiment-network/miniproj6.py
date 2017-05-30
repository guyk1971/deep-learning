# Mini project 6
# same as miniproj 5 but with further noise reduction
"""
Modify pre_process_data:
------------------------
1. Add two additional parameters: min_count and polarity_cutoff
2. Calculate the positive-to-negative ratios of words used in the reviews. (You can use code you've written elsewhere in the notebook, but we are moving it into the class like we did with other helper code earlier.)
3. Andrew's solution only calculates a postive-to-negative ratio for words that occur at least 50 times. This keeps the network from attributing too much sentiment to rarer words. You can choose to add this to your solution if you would like.
4. Change so words are only added to the vocabulary if they occur in the vocabulary more than min_count times.
5. Change so words are only added to the vocabulary if the absolute value of their postive-to-negative ratio is at least polarity_cutoff

Modify __init__:
----------------
1. Add the same two parameters (min_count and polarity_cutoff) and use them when you call pre_process_data
"""
import time
import sys
from collections import Counter
import numpy as np


# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, min_count=10, polarity_cutoff=0.1, hidden_nodes=10, learning_rate=0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training

        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training

        self.pre_process_data(reviews, labels, min_count=min_count, polarity_cutoff=polarity_cutoff)

        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels, min_count, polarity_cutoff):

        positive_counts=Counter()
        negative_counts=Counter()
        total_counts=Counter()
        for indx in range(len(labels)):
            if labels[indx] == "POSITIVE":
                positive_counts.update(reviews[indx].split(' '))
            else:
                negative_counts.update(reviews[indx].split(' '))
            total_counts.update(reviews[indx].split(' '))


        # Calculate the ratios of positive and negative uses of the most common words
        #       Consider words to be "common" if they've been used at least 100 times
        self.pos_neg_ratios = Counter()
        for word in positive_counts.keys():
            if positive_counts[word] > 49:
                self.pos_neg_ratios[word] = positive_counts[word] / float(negative_counts[word] + 1)

        for word in self.pos_neg_ratios.keys():
            ratio = self.pos_neg_ratios[word]
            self.pos_neg_ratios[word] = np.log(ratio) if ratio > 1 else -np.log(1 / (ratio + 0.01))


        review_vocab = set()
        # populate review_vocab with all of the words in the given reviews
        #       miniproj6 - add them only if they occur more than min_count times
        #       Remember to split reviews into individual words
        #       using "split(' ')" instead of "split()".
        # Answer:
        for review in reviews:
            words = review.split(" ")
            for w in words:
                if total_counts[w] >= min_count and np.abs(self.pos_neg_ratios[w])>polarity_cutoff:
                    review_vocab.add(w)

        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)

        label_vocab = set()
        # populate label_vocab with all of the words in the given labels.
        #       There is no need to split the labels because each one is a single word.
        # Answer:
        for label in labels:
            label_vocab.add(label)

        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)

        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        # populate self.word2index with indices for all the words in self.review_vocab
        #       like you saw earlier in the notebook
        # Answer:
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        # do the same thing you did for self.word2index and self.review_vocab,
        #       but for self.label2index and self.label_vocab instead
        # Answer:
        for i, word in enumerate(self.label_vocab):
            self.label2index[word] = i



    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights

        # initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.

        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

        #  initialize self.weights_1_2 as a matrix of random values.
        #       These are the weights between the hidden layer and the output layer.

        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, (self.hidden_nodes, self.output_nodes))

        # remove layer_0 and replace it with layer_1
        #       1 x hidden_nodes, with all values initialized to zero
        # Answer:
        self.layer_1 = np.zeros((1, hidden_nodes))

    def get_target_for_label(self, label):
        # Copy the code you wrote for get_target_for_label
        #       earlier in this notebook.
        # Answer:
        return 1 if label == 'POSITIVE' else 0

    def sigmoid(self, x):
        # Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        # Answer:
        return 1./(1.+np.exp(-x))

    def sigmoid_output_2_derivative(self, output):
        # Return the derivative of the sigmoid activation function,
        #       where "output" is the original output from the sigmoid fucntion
        # Answer:
        return output*(1-output)

    def train(self, training_reviews_raw, training_labels):

        # make sure out we have a matching number of reviews and labels
        assert (len(training_reviews_raw) == len(training_labels))

        # convert training reviews to list of indices 
        training_reviews=[]
        for i in range(len(training_reviews_raw)):
            indxset=set()
            for w in training_reviews_raw[i].split(" "):
                if w in self.word2index.keys():
                    indxset.add(self.word2index[w])
            training_reviews.append(list(indxset))
            
        
        
        # Keep track of correct predictions to display accuracy during training
        correct_so_far = 0

        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):

            # Get the next review and its correct label
            review_i = training_reviews[i]
            label_i = training_labels[i]

            # Implement the forward pass through the network.
            #       That means use the given review to update the input layer,
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            #
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.
            self.layer_1 *= 0
            for indx in review_i:
                self.layer_1+=self.weights_0_1[indx]
            
            output_layer = self.sigmoid(self.layer_1.dot(self.weights_1_2))

            # Implement the back propagation pass here.
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you
            #       learned in class.
            # Answer:
            error = self.get_target_for_label(label_i) - output_layer
            grad_err_output = error * self.sigmoid_output_2_derivative(output_layer)    # shape = [batch_size,1]
            dW_1_2 = self.layer_1.T.dot(grad_err_output)

            grad_err_hidden = grad_err_output.dot(self.weights_1_2.T)   # shape = [batch_size,hidden_units]

            self.weights_1_2 += self.learning_rate * dW_1_2
            for indx in review_i:
                self.weights_0_1[indx] += self.learning_rate * grad_err_hidden[0]

            # TODO: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output error
            #       is less than 0.5. If so, add one to the correct_so_far count.
            # Answer:
            if np.abs(error)<0.5:
                correct_so_far += 1

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the training process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i + 1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i + 1))[:4] + "%")
            if (i % 2500 == 0):
                print("")

    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """

        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label.
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if (pred == testing_labels[i]):
                correct += 1

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the prediction process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i + 1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i + 1))[:4] + "%")

    def run(self, review_raw):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # TODO: Run a forward pass through the network, like you did in the
        #       "train" function. That means use the given review to
        #       update the input layer, then calculate values for the hidden layer,
        #       and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction
        #             might come from anywhere, so you should convert it
        #             to lower case prior to using it.
        indxset=set()
        for w in review_raw.split(" "):
            if w in self.word2index.keys():
                indxset.add(self.word2index[w])
        review=list(indxset)

        self.layer_1 *= 0
        for indx in review:
            self.layer_1+=self.weights_0_1[indx]
        
        output_layer = self.sigmoid(self.layer_1.dot(self.weights_1_2))


        # TODO: The output layer should now contain a prediction.
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`,
        #       and `NEGATIVE` otherwise.
        return 'POSITIVE' if output_layer >= 0.5 else 'NEGATIVE'



if __name__=='__main__':
    g = open('reviews.txt', 'r')  # What we know!
    reviews = list(map(lambda x: x[:-1], g.readlines()))
    g.close()

    g = open('labels.txt', 'r')  # What we WANT to know!
    labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
    g.close()
    # create the mlp
#    mlp = SentimentNetwork(reviews[:-1000], labels[:-1000], learning_rate=0.1)

    # test before training
#    mlp.test(reviews[-1000:], labels[-1000:])

    # now train with the lr=0.1 (too high)
#    mlp.train(reviews[:-1000], labels[:-1000])

    # reduce the learning rate to 0.01 and try again
#    mlp = SentimentNetwork(reviews[:-1000], labels[:-1000], learning_rate=0.01)
#    mlp.train(reviews[:-1000], labels[:-1000])

    # now do it with a learning rate that should work
    mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], min_count=20, polarity_cutoff=0.05, learning_rate=0.001)
    mlp.train(reviews[:-1000],labels[:-1000])
    mlp.test(reviews[-1000:], labels[-1000:])
