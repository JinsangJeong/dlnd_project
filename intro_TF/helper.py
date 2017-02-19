import math

def batches(batch_size, features, labels):

	assert len(features) == len(labels)

	outout_batches = []

	sample_size = len(features)

	for start_i in range(0, sample_size, batch_size):
		end_i = start_i + batch_size
		batch = [features[start_i:end_i], labels[start_i:end_i]]
		outout_batches.append(batch)

	return outout_batches