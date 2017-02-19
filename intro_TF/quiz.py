import tensorflow as tf 

def get_weights(n_featrues, n_labels):
	return tf.Variable(tf.truncated_normal((n_featrues, n_labels)))

def get_biases(n_labels):
    return tf.Variable(tf.zeros(n_labels))

def linear(input, w, b):
    return tf.add(tf.matmul(input, w), b)