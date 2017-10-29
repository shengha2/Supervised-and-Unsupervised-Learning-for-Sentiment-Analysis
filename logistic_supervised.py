import os
import re
import random
import math
import numpy as np
import matplotlib.pyplot as plt



def get_reviews(directory,size_train,size_test):
    train_reviews = []
    test_reviews = []
    words = set()
    filenames = os.listdir(directory)
    random.shuffle(filenames)
    if size_train+size_test > len(filenames):
        print('Train, test, and validaition size too large.')
        return
    for filename in filenames[:size_train]:
        review = re.sub(r'[^\w\s]',' ',open(directory+filename).read().lower()).split()
        train_reviews.append(set(review))
        words = words.union(set(review))
    for filename in filenames[size_train:size_train+size_test]:
        review = re.sub(r'[^\w\s]',' ',open(directory+filename).read().lower()).split()
        test_reviews.append(set(review))
        words = words.union(set(review))
    return (train_reviews, test_reviews, words)

def get_data_sets(num_train, num_test):
    neg_review = 'txt_sentoken/Neg/'
    pos_review = 'txt_sentoken/Pos/'
    all_words = set()
    (neg_review_train,neg_review_test, neg_words) = get_reviews(neg_review,num_train,num_test)
    (pos_review_train,pos_review_test, pos_words) = get_reviews(pos_review,num_train,num_test)
    all_words = all_words.union(neg_words)
    all_words = all_words.union(pos_words)
    all_words = list(all_words)
    print "Aquiring training data..."
    train_data, train_classifier = make_matrix(all_words, pos_review_train, neg_review_train)
    print "Aquired training data"
    print "Aquiring testing data..."
    test_data, test_classifier = make_matrix(all_words, pos_review_test, neg_review_test)
    print "Aquired testing data"
    return train_data, train_classifier, test_data, test_classifier, all_words

def make_matrix(all_words, pos_review, neg_review):
    num_reviews = len(pos_review)+len(neg_review)
    num_words = len(all_words)
    reviews = pos_review + neg_review
    data = np.zeros((num_reviews, num_words))
    classifier_positive = np.ones((len(pos_review),1))
    classifier_negative = np.zeros((len(neg_review),1))
    classifier = np.concatenate((classifier_positive, classifier_negative), axis=0)
    for i in range(len(reviews)):
        for j in range(len(all_words)):
            if all_words[j] in reviews[i]:
                data[i][j] = 1
    bias = np.ones((num_reviews, 1))
    data = np.concatenate((bias, data), axis=1)
    return data, classifier

################################################
#Logistic Regression
################################################

def logreg_softmax(weights, images):
	'''
	Function: generate softmax probabilities
	Input:
		weights: matrix of weights of size (785 x 10)
		images: matrix of flattened images as column vectors (785 x #images)
	Output:
		probability: probability for classifying as each digit for each image is stored as a column vector (10 x #images)
	'''
	guess = get_classification(weights, images)
	probability = np.exp(guess)
	probability = probability/probability.sum(axis=0)[None,:]
	return probability

def get_classification(weights, images):
	classification = np.dot(images, weights)
	return classification

def get_sigmoid(weights, images):
    classification = get_classification(weights, images)
    sigmoid = 1/(1+np.exp(-classification))
    return sigmoid

def logreg_get_cost_matrix(weights, images, classifier):
	sigmoid = get_sigmoid(weights, images)
	cost_matrix = sigmoid - classifier
	cost_matrix = np.square(cost_matrix)
	return cost_matrix

def logreg_get_cost(weights, images, classifier, lambda_value):
	cost_matrix = logreg_get_cost_matrix(weights, images, classifier)
	cost = np.sum(cost_matrix) + ((lambda_value*np.sum(weights))**2)/2
	return cost

def logreg_get_gradient(weights, images, classifier, lambda_value):
	sigmoid = get_sigmoid(weights, images)
	# print sigmoid
	error = sigmoid-classifier
	# print error
	gradient = np.dot(images.T,(error*sigmoid*(1-sigmoid))) + (lambda_value*weights)
	# print gradient
	return gradient

def logreg_grad_descent(weights,images,classifier,iterations, get_history, step_size, lambda_value):
	'''
	Function: runs a gradient descent to minimize the cost function by updating the weight matrix
	Input:
		weights: intial guess of classifier function (matrix or vector)
		images: image data used to calculate output of classifier function (matrix)
		classifier: target outputs also used to calculate output of classifier function (matrix)
		iterations: number of iterations to run for the gradient descent
		get_history: Boolean to determine whether to save history of weights
		step_size: how frequent to save history of weights
	Output:
		old_weights: the weight matrix after gradient descent
		history: a list of the history of weights
	'''
	alpha = 0.000354483954405# 0.1
	old_weights = weights
	current_cost = logreg_get_cost(old_weights, images, classifier, lambda_value)
	i = 0
	initial_cost = current_cost
	initial_weights = old_weights
	history = [old_weights]
	while iterations>=i:
		if i%step_size == 0 and i !=0: history.append(old_weights)
		if i%100 == 0 and i != 0: print "iteration:", i, "cost:", current_cost
		gradient = logreg_get_gradient(old_weights, images, classifier, lambda_value)
		new_weights = old_weights - alpha*gradient
		new_cost = logreg_get_cost(new_weights, images, classifier, lambda_value)
		if current_cost < new_cost or math.isnan(new_cost):
			current_cost = initial_cost
			old_weights = initial_weights
			alpha = alpha*0.95
			history = [initial_weights]
			i = 0
		else:
			old_weights = new_weights
			current_cost = new_cost
			i+=1
	if get_history == True:
		return old_weights, history
	else:
		return old_weights

def accuracy(weights, data, classifier):
    pred = get_classification(weights, data)
    correct = 0
    for i in range(len(classifier)):
        if pred[i]<0.5 and classifier[i] == 0:
            correct +=1
        if pred[i]>0.5 and classifier[i] == 1:
            correct +=1
    return float(correct)/float(len(classifier))

def part1(test_words):
    print "Part 1"
    neg_review = 'txt_sentoken/Neg/'
    pos_review = 'txt_sentoken/Pos/'
    num_train = 800
    num_test = 200
    (neg_review_train,neg_review_test, neg_words) = get_reviews(neg_review,num_train,num_test)
    (pos_review_train,pos_review_test, pos_words) = get_reviews(pos_review,num_train,num_test)
    for word in test_words:
        positive = 0
        negative = 0
        for review in neg_review_train:
            if word in review: negative +=1
        for review in neg_review_test:
            if word in review: negative +=1
        for review in pos_review_train:
            if word in review: positive +=1
        for review in pos_review_test:
            if word in review: positive +=1
        print "\n"
        print "Word:", word
        print "Number of Positive Reviews:", positive
        print "Number of Negative Reviews:", negative
        print "Percentage of Positive Reviews:", float(positive)/10.0, "%"
        print "Percentage of Negative Reviews:",float(negative)/10.0, "%"
    print "\n"
    return

def part4(num_iterations, train_data, train_classifier, test_data, test_classifier, all_words, plot_step_size):
    print "part 4"
    lambda_values = [0, 0.1, 0.2, 0.3, 0.4]
    num_words = len(all_words)
    weights = (np.random.rand(num_words+1,1)-0.5)/5
    weights_0, history_0 = logreg_grad_descent(weights,train_data,train_classifier,num_iterations, True, plot_step_size, lambda_values[0])
    weights_1, history_1 = logreg_grad_descent(weights,train_data,train_classifier,num_iterations, True, plot_step_size, lambda_values[1])
    weights_2, history_2 = logreg_grad_descent(weights,train_data,train_classifier,num_iterations, True, plot_step_size, lambda_values[2])
    weights_3, history_3 = logreg_grad_descent(weights,train_data,train_classifier,num_iterations, True, plot_step_size, lambda_values[3])
    weights_4, history_4 = logreg_grad_descent(weights,train_data,train_classifier,num_iterations, True, plot_step_size, lambda_values[4])
    test_performance_0 = []
    test_performance_1 = []
    test_performance_2 = []
    test_performance_3 = []
    test_performance_4 = []
    bad_words = []
    good_words = []
    iterations = np.arange(num_iterations/plot_step_size+1)*plot_step_size
    weights_0 = np.reshape(weights_0, (weights_0.shape[1], weights_0.shape[0]))[0]
    weights_0 = weights_0[1:]
    abs_weights = abs(weights_0)
    indicies_important_words =  abs_weights.argsort()[-100:][::-1]
    print indicies_important_words
    for i in range(len(indicies_important_words)):
        index = indicies_important_words[i]
        if weights_0[index] < 0:
            bad_words.append(all_words[index])
        else:
            good_words.append(all_words[index])
    print bad_words
    print good_words
    for i in range(len(history_0)):
        test_performance_0.append(accuracy(history_0[i], test_data, test_classifier))
        test_performance_1.append(accuracy(history_1[i], test_data, test_classifier))
        test_performance_2.append(accuracy(history_2[i], test_data, test_classifier))
        test_performance_3.append(accuracy(history_3[i], test_data, test_classifier))
        test_performance_4.append(accuracy(history_4[i], test_data, test_classifier))
    plt.suptitle('Performance of Logistic Regression Test Set for Different Lambdas')
    plt.xlabel('Iterations')
    plt.ylabel('Percent Correctly Classified')
    plt.plot(iterations, test_performance_0, 'b',label="Lambda = 0")
    plt.plot(iterations, test_performance_1, 'y',label="Lambda = 0.1")
    plt.plot(iterations, test_performance_2, 'r',label="Lambda = 0.2")
    plt.plot(iterations, test_performance_3, 'g',label="Lambda = 0.3")
    plt.plot(iterations, test_performance_4, '-',label="Lambda = 0.4")
    plt.legend(loc='best')
    plt.show()
    return


train_size = 800
test_size = 200
train_data, train_classifier, test_data, test_classifier, all_words = get_data_sets(train_size, test_size)
num_iterations = 1000
plot_step_size = 10
test_words = ["bad", "worst", "hilarious"]
part1(test_words)
part4(num_iterations, train_data, train_classifier, test_data, test_classifier, all_words, plot_step_size)
