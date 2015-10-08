import numpy
from numpy import linalg, random as rn
import matplotlib.pyplot as plt
import sys
import random

def load_data(filename):
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
	print "Number of errors is " + str(errors) + " out of " + str(num_samples)

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

def SGD(dataset, stepsize_constant, lambda_):
	print "Starting SGD algorithm with stepsize_constant: " + str(stepsize_constant) + " lambda: " + str(lambda_)
	random.shuffle(dataset)
	init = rn.normal(size=(1, len(dataset[0][0])))[0]
	w = scale_w(lambda_, init)
	errors = []
	errors.append(total_error(w, lambda_, dataset))
	for i in range(0, len(dataset)):
		datum = dataset[i]
		# Stepsize is current c / sqrtroot(t)
		stepsize = float(stepsize_constant)/ numpy.sqrt((float(i + 1)))
		delta = derivative(datum, w) + lambda_ * numpy.array(w)
		delta = stepsize * numpy.array(delta)
		w = w - delta
		w = scale_w(lambda_, w)
		if i < 100 :
			#Bottle neck right now
			errors.append(total_error(w, lambda_, dataset))
	print "Final w from SGD is " + str(w)
	return (w, errors)


def main():
	if len(sys.argv) != 2:
		print "Please enter a file"
		return
	filename = sys.argv[1]	
	(wstar, data) = load_data(filename)
	(result, errors) = SGD(data, 1, 1)
	count_errors(result, data)
	plt.plot(errors)
	plt.ylabel('Errors')
	plt.show()

main()