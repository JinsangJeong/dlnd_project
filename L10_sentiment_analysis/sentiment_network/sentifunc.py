'''Project_1_Curate a dataset'''

# def pretty_print_review_and_label(i):
# 	print(labels[i] + "\t:\t" + reviews[i][:80] + '...')

# g = open('reviews.txt', 'r') # What we know
# reviews = list(map(lambda x: x[:-1], g.readlines()))
# g.close()

# g = open('labels.txt', 'r') #What we WANT to know
# labels = list(map(lambda x: x[:-1], g.readlines()))
# g.close()

# print('labels.txt \t:\t reviews.txt\n')
# # pretty_print_review_and_label(234)
# # pretty_print_review_and_label(456)
# # pretty_print_review_and_label(1367)

# '''Quick Theory Validation'''
# from collections import Counter
# import numpy as np

# positive_counts = Counter()
# negative_counts = Counter()
# total_counts = Counter()

# for i in range(len(reviews)):
# 	if labels[i] == 'POSITIVE':
# 		for word in reviews[i].split(' '):
# 			positive_counts[word] += 1
# 			total_counts[word] += 1
# 	else:
# 		for word in reviews[i].split(' '):
# 			negative_counts[word] += 1
# 			total_counts[word] += 1

# print(positive_counts.most_common())
# print(negative_counts.most_common())

# pos_neg_ratios = Counter()

# for term, cnt in total_counts.most_common():
#     if cnt >10:
#     	pos_neg_ratio = positive_counts[term]/(negative_counts[term] + 1)
#     	pos_neg_ratios[term] = pos_neg_ratio

# for word, ratio in pos_neg_ratios.most_common():
# 	if ratio > 1:
# 		pos_neg_ratios[word] = np.log(ratio)
# 	else:
# 		pos_neg_ratios[word] = -np.log(1/(ratio + 0.01))

#Project_2_Transforming Text into Numbers

import numpy as np

def pre_process_data(reviews):
    review_vocab = set()
    for review in reviews:
    	for word in review.split(' '):
    		review_vocab.add(word)

    review_vocab = list(review_vocab)
    input_nodes = len(review_vocab)
    
    word2index = {}
    for i, word in enumerate(review_vocab):
        word2index[word] = i 

    return word2index, input_nodes

def update_input_layer(review, word2index, layer_0):
    # clear out the previous state, reset the layer to be all 0
    layer_0 *= 0

    for word in review.split(' '):
    	layer_0[0][word2index[word]] += 1

    return layer_0

def get_target_for_label(label):
	if label == 'POSITIVE':
		return 1
	else:
		return 0 

def sigmoid(x):
	return 1 / (1 + np.exp(-x))













