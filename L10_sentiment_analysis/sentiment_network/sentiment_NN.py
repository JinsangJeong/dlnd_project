import sys
import time
import sentifunc as sf 
import numpy as np

class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 100, \
    	output_nodes = 1, learning_rate=0.01):

        # set our random number generator
        np.random.seed(1)

        self.word2index, self.input_nodes = sf.pre_process_data(reviews)
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        self.layor_0 = np.zeros((1, self.input_nodes))

        self.sigmoid = lambda x : 1 / (1 + np.exp(-x))

    def train(self, training_reviews, training_labels):
    	assert(len(training_reviews) == len(training_labels))

    	correct_so_far = 0

    	start = time.time()

    	for i in range(len(training_reviews)):
            reveiw = reviews[i]
            label = labels[i]

        	## Inplement the forward pass here
        	# Forward pass #

        	# Input Layor
            self.layor_0 = sf.update_input_layer(reveiw, self.word2index, self.layor_0)    

            # Hidden Layor
            layor_1 = np.dot(self.layor_0, self.weights_0_1)

            # Output Layor
            output = self.sigmoid(np.dot(layor_1, self.weights_1_2))

            ## Implement the backward pass here
            # Backward here #

            # Output error
            output_error = sf.get_target_for_label(label) - output 
            output_e_delta = np.dot(output_error, output*(1-output))

            # Backpropagated error
            hidden_error = np.dot(output_e_delta, self.weights_1_2.T)
            hidden_e_delta = np.dot(hidden_error, 1)

            # Update the weights
            self.weights_1_2 += self.lr*output_e_delta*layor_1.T
            self.weights_0_1 += self.lr*hidden_e_delta*self.layor_0.T

            if np.abs(output_error) < 0.5:
            	correct_so_far += 1

            reveiws_per_second = i / float(time.time() - start)
            if i % 1000 == 0:
                print('Progress:' + str(100*(i/len(training_reviews)))[:4] + '% Speed(reveiws/sec):' + str(reveiws_per_second)[:5] + ', Correct:' + str(correct_so_far) + ', Trained:' + str(i+1) + ', Training Accuracy:' + str(correct_so_far*100/float(i+1))[:4] + '%')
            # if i % 2500 == 0:
            # 	print('')

    def test(self, testing_reviews, testing_labels):
    	correct_so_far = 0

    	start = time.time()

    	for i in range(len(testing_reviews)):
    		pred = self.run(testing_reviews[i])
    		if pred == testing_labels[i]:
    			correct_so_far += 1

    		reveiws_per_second = i / float(time.time() - start)
    		if i % 1000 == 0:
    		    print('Progress:' + str(100*(i/len(testing_reviews)))[:4] + '% Speed(reveiws/sec):' + str(reveiws_per_second)[:5] + ', Correct:' + str(correct_so_far) + ', Tested:' + str(i+1) + ', Testing Accuracy:' + str(correct_so_far*100/float(i+1))[:4] + '%')

    def run(self, review):

    	# Input Layor
        self.layor_0 = sf.update_input_layer(review, self.word2index, self.layor_0)    

        # Hidden Layor
        layor_1 = np.dot(self.layor_0, self.weights_0_1)

        # Output Layor
        output = self.sigmoid(np.dot(layor_1, self.weights_1_2))

        if output[0] < 0.5:
        	return 'POSITIVE'
        else:
        	return 'NEGATIVE'

if __name__ == '__main__':
	g = open('reviews.txt', 'r')
	reviews = list(map(lambda x:x[:-1], g.readlines()))
	g.close()

	g = open('labels.txt', 'r')
	labels = list(map(lambda x:x[:-1].upper(), g.readlines()))
	g.close()

	mpl = SentimentNetwork(reviews[:-1000], labels[:-1000], learning_rate=0.01)

	# evaluate our model before training (just show how horrible it is)
	#mpl.test(reviews[:-1000], labels[:-1000])

	# train the network
	mpl.train(reviews[:-1000], labels[:-1000])

