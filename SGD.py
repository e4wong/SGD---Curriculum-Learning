import numpy
from numpy import linalg, random as rn
import matplotlib.pyplot as plt
import sys
import random
import math

x = []
y = []

def load_data(filename):
	global x
	global y
	f = open(filename, "r")
	print "Reading from file " + filename
	# Data from the UniformDataGenerator.py has the format
	# such that the first line is w*
	line = f.readline()
	line = line[line.index('[') + 1 : len(line) - 2]
	# Remove the other comment part of the first line of w*
	tokens = line.split()
	wstar = []
	for token in tokens:
		wstar.append(float(token))

	print "W* is " + str(wstar)

	ds = []
	for line in f:
		features = []
		sign = 0
		tokens = line.split()
		for i in range(0, len(tokens) - 1):
			features.append(float(tokens[i]))
		sign = int(tokens[len(tokens) -1])
		x.append(features)
		y.append(sign)
		ds.append((features,sign))
	print "Done loading data"
	return (wstar,ds)

def count_errors(wstar, dataset):
	errors = 0
	num_samples = len(dataset)
	for (features, label) in dataset:
		dp = numpy.dot(features, wstar)
		if dp > 0 and label == -1:
			errors = errors + 1
		elif dp < 0 and label == 1:
			errors = errors + 1
		elif dp == 0:
			# just consider this an error, right on decision boundary
			errors = errors + 1
	return errors

def total_error_matrix_optimize(w, lambda_, dataset):
	global x
	global y
	x = numpy.matrix(x)
	w_ = numpy.matrix(w)
	wTx = x * w_.T
	wTx = numpy.squeeze(numpy.asarray(wTx))
	e_neg_wTx_y = [numpy.log(1 + numpy.exp(-1 * wTx[i] * y[i])) for i in range(0, len(wTx))]
	return sum(e_neg_wTx_y)/len(e_neg_wTx_y) + (float(lambda_)/2.0) * linalg.norm(w) ** 2


def total_error(w, lambda_, dataset):
	sum_errors = float(0)
	for datum in dataset:
		sum_errors += logistic_loss(datum, w)
	sum_errors /= len(dataset)
	return sum_errors + (float(lambda_) / 2.0) * linalg.norm(w) ** 2

def logistic_loss(datum, w):
	(x, y) = datum
	xp = -1 * y * numpy.dot(w, x)  
	return numpy.log(1 + numpy.exp(xp))

def scale_w(lambda_, w):
	mag = float(1)/ float(lambda_)
	if linalg.norm(w) > mag:
		w = (1.0/linalg.norm(w)) * (1.0/float(lambda_))  * numpy.array(w)
	return w

def derivative(datum, w):
	(x, y) = datum
	deno = 1.0 + numpy.exp(y * numpy.dot(x, w))
	return  ((-float(y))/deno) * numpy.array(x) 

def stepsize_fn(c, lambda_, t):
	config = 1
	if config == 1:
		return float(c) / float(lambda_ * t)
	else:
		return float(c) / numpy.sqrt(t)
def SGD(training_set, stepsize_constant, lambda_, error_log, validation_set):
	print "Starting SGD algorithm with stepsize_constant: " + str(stepsize_constant) + " lambda: " + str(lambda_)
	init = rn.normal(size=(1, len(training_set[0][0])))[0]
	w = scale_w(lambda_, init)
	errors = []
	errors.append(total_error_matrix_optimize(w, lambda_, validation_set))
	for i in range(0, len(training_set)):
		datum = training_set[i]
		# Stepsize is current c / sqrtroot(t)
		stepsize = stepsize_fn(stepsize_constant, lambda_, i + 1)
		delta = derivative(datum, w) + lambda_ * numpy.array(w)
		delta = stepsize * numpy.array(delta)
		w = w - delta
		w = scale_w(lambda_, w)
		if error_log and i % len(training_set) / 1000 == 0 :
			#Bottle neck right now
			errors.append(total_error_matrix_optimize(w, lambda_, validation_set))
	print "Final w from SGD is " + str(w)
	return (w, errors)

def run_SGD(training_set, validation_set, stepsize_constant):
	# configurations of lambda
	exponents = [-3, -2, -1, 0]
	base = 10
	
	#defaults to compare to
	best_mistakes = len(validation_set)
	best_lambda = 1
	for exp in exponents:
		curr_lambda = float(math.pow(base, exp))
		(w, errors) = SGD(training_set, stepsize_constant, curr_lambda, False, validation_set) 
		mistakes = count_errors(w, validation_set)
		if mistakes < best_mistakes:
			best_mistakes = mistakes
			best_lambda = curr_lambda
	return SGD(training_set, stepsize_constant, best_lambda, True, validation_set)

def main():
	if len(sys.argv) != 2:
		print "Please enter a file"
		return
	filename = sys.argv[1]	
	(wstar, data) = load_data(filename)

	random.shuffle(data)

	training_set = data[len(data)/2 : ]
	validation_set = data[ : len(data)/2]
	
	(result, errors) = run_SGD(training_set, validation_set, 1)
	print "Validation_set Accuracy: " + str(float(count_errors(result, validation_set))/len(validation_set))
	plt.plot(errors)
	plt.ylabel('Errors')
	plt.show()

main()