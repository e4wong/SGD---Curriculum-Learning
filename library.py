import numpy
from numpy import linalg, random as rn

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

def print_dataset_general(wstar, labeled_data):
    wstarconfig =  "//w* = " + str(wstar)
    wstarconfig = wstarconfig.replace("\n", "")
    wstarconfig = wstarconfig.replace(",", "")
    print wstarconfig
    for (data,label) in labeled_data:
        string = ""
        for dimension in data:
            string += str(dimension) + " "
        string += str(label)
        print string


def load_data_no_print(filename):
    f = open(filename, "r")
    # Data from the UniformDataGenerator.py has the format
    # such that the first line is w*
    line = f.readline()
    line = line[line.index('[') + 1 : len(line) - 2]
    # Remove the other comment part of the first line of w*
    tokens = line.split()
    wstar = []
    for token in tokens:
        wstar.append(float(token))


    ds = []
    for line in f:
        features = []
        sign = 0
        tokens = line.split()
        for i in range(0, len(tokens) - 1):
            features.append(float(tokens[i]))
        sign = int(tokens[len(tokens) -1])
        ds.append((features,sign))
    return (wstar,ds)

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

def derivative(datum, w):
	(x, y) = datum
	deno = 1.0 + numpy.exp(y * numpy.dot(x, w))
	return  ((-float(y))/deno) * numpy.array(x) 


default_output_file = "output"
def output_final_w(wstar):
	f = open(default_output_file,'w')
	wstarconfig = "//w* is " + str(wstar)
	wstarconfig = wstarconfig.replace("\n", "")
	wstarconfig = wstarconfig.replace(",", "")
	f.write(wstarconfig)

def calc_error_rate(wstar, dataset):
	return float(count_errors(wstar, dataset))/ float(len(dataset))

def count_errors(wstar, dataset):
	errors = 0
	num_samples = len(dataset)
	for (features, label) in dataset:
		dp = numpy.dot(features, wstar)
		if dp > 0 and label == -1:
			errors = errors + 1
		elif dp < 0 and label == 1:
			errors = errors + 1
		elif dp == 0 and label == 1:
			# <= 0 -> -1, so 1 would be an error
			errors = errors + 1
	return errors
