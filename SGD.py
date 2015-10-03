import numpy
from numpy import linalg
import sys

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
	return sum_errors + (lambda_ / 2) * linalg.norm(w) ** 2

def logistic_loss(datum, w):
	(x, y) = datum
	xp = -1 * y * numpy.dot(w, x)  
	return numpy.log(1 + numpy.exp(xp))

def main():
	if len(sys.argv) != 2:
		print "Please enter a file"
		return
	filename = sys.argv[1]	
	(wstar, data) = load_data(filename)


main()