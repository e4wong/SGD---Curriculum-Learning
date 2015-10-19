import numpy
from numpy import linalg, random as rn
import matplotlib.pyplot as plt
import sys
import random
import math
import copy
from sklearn.linear_model import SGDClassifier

x = []
y = []

exponents = [-2, -1, 0]
base = 10

stepsize_constant_var = 1

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
	# if lambda = .001 , then 1/.001 causes overflow when we do e^ in derivative
	mag = float(1)/ float(lambda_)
	if linalg.norm(w) > mag:
		w = (1.0/linalg.norm(w)) * (1.0/float(lambda_))  * numpy.array(w)
	return w

def derivative(datum, w):
	(x, y) = datum
	deno = 1.0 + numpy.exp(y * numpy.dot(x, w))
	return  ((-float(y))/deno) * numpy.array(x) 

def dotproduct(v1, v2):
 	return sum((a*b) for a, b in zip(v1, v2))

def length(v):
	return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
	return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def cos(val):
	return math.cos(val)

def stepsize_fn(c, lambda_, t):
	config = 1
	if config == 1:
		return float(c) / float(lambda_ * t)
	else:
		return float(c) / numpy.sqrt(t)

def SGD(training_set, stepsize_constant, lambda_, error_log, validation_set):
	#print "Starting SGD algorithm with stepsize_constant: " + str(stepsize_constant) + " lambda: " + str(lambda_)
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
		if error_log and (i % 100) == 0:
			#Bottle neck right now
			errors.append(total_error_matrix_optimize(w, lambda_, validation_set))
	#print "Final w from SGD is " + str(w) + "\n"
	return (w, errors)

def find_lambda(training_set, validation_set, stepsize_constant):
	# configurations of lambda
	global exponents
	global base 
	
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
	return best_lambda

def run_SGD(training_set, validation_set, stepsize_constant, plot):
	best_lambda = find_lambda(training_set,validation_set, stepsize_constant)
	return SGD(training_set, stepsize_constant, best_lambda, plot, validation_set)

def sort_data(sort_by, data, w_star):
	# sanity check, should always be hard or easy
	if not(sort_by == "hard") and not(sort_by == "easy") and not(sort_by == "random"):
		print "Something went wrong!!!!!!!"
		return [] 

	data_hardness = []
	for (features, label) in data:
		cos_val = cos(angle(w_star,features))
		data_hardness.append(((features,label), abs(cos_val)))

	if not(sort_by == "random"):
		data_hardness.sort(key=lambda tup: tup[1])

	# Hard means that hard examples come first, which means the lower abs(cos_val) are first in the list
	# Easy means that easy examples come first, which means the higher abs(cos_val) are first in the list
	if sort_by == "easy":
		data_hardness.reverse()
	train = [data for (data, hardness) in data_hardness]	
	return train

def curriculum_learning(sort_by, times_to_run, original_data, w_star, lambda_):
	global stepsize_constant_var
	error_rate = []
	abs_cos_val = []

	for i in range(0, times_to_run):
		data = copy.deepcopy(original_data)
		random.shuffle(data)
		training_set = data[len(data)/2 : ]
		validation_set = data[ : len(data)/2]
		training_set = sort_data(sort_by, training_set, w_star)
		(w, errors) = SGD(training_set, stepsize_constant_var, lambda_, False, validation_set)
		error_rate.append(calc_error_rate(w, validation_set))
		abs_cos_val.append(cos(angle(w,w_star)))
	return (error_rate, abs_cos_val)

def trace_objective_function_CL(sort_by, data, lambda_, w_star):
	random.shuffle(data)
	training_set = data[len(data)/2 : ]
	validation_set = data[ : len(data)/2]
	training_set = sort_data(sort_by, training_set, w_star)
	(w, obj_plot) = SGD(training_set, stepsize_constant_var, lambda_, True, validation_set)
	return obj_plot

def main():
	if len(sys.argv) < 2:
		print "Please enter a file"
		return
	elif len(sys.argv) == 2:
		filename = sys.argv[1]	
		(wstar, data) = load_data(filename)
		random.shuffle(data)

		training_set = data[len(data)/2 : ]
		validation_set = data[ : len(data)/2]

		(result, errors) = run_SGD(training_set, validation_set, stepsize_constant_var, True)
		print "Validation Set Error Rate: " + str(calc_error_rate(result, validation_set))
		output_final_w(result)
		plt.plot(errors)
		plt.ylabel('Objective Function')
		plt.show()

	elif len(sys.argv) == 3:
		filename = sys.argv[1]	
		(wstar, data) = load_data(filename)

		num_runs = int(sys.argv[2])		
		print "Curriculum learning, Number of iterations for both hard and easy to run:", num_runs 
		
		random.shuffle(data)
		training_set = data[len(data)/2 : ]
		validation_set = data[ : len(data)/2]
		lambda_ = find_lambda(training_set,validation_set, stepsize_constant_var)
		print "Running Hard->Easy"
		(hard_error_rate, hard_cos_val) = curriculum_learning("hard", num_runs, data, wstar, lambda_)
		print "Running Easy->Hard"
		(easy_error_rate, easy_cos_val) = curriculum_learning("easy", num_runs, data, wstar, lambda_)
		print "Running Random"
		(random_error_rate, random_cos_val) = curriculum_learning("random", num_runs, data, wstar, lambda_)
		
		print "Error Rate of w*:", calc_error_rate(wstar, data)

		plt.figure(0)
		plt.plot(hard_error_rate)
	 	plt.plot(easy_error_rate)
	 	plt.plot(random_error_rate)
		plt.legend(['Hard Examples First', 'Easy Examples First', 'Normal/Random Examples'], loc='upper left')
		plt.ylabel('Error Rate')

		plt.figure(1)
		plt.plot(hard_cos_val)
		plt.plot(easy_cos_val)
		plt.plot(random_cos_val)
		plt.legend(['Hard Examples First', 'Easy Examples First', 'Normal/Random Examples'], loc='lower right')
		plt.ylabel("Abs(Cos(w,w*))")
	
		print "Tracing Hard->Easy"
		obj_plot_hard = trace_objective_function_CL("hard", data, lambda_, wstar)
		print "Tracing Easy->Hard"
		obj_plot_easy = trace_objective_function_CL("easy", data, lambda_, wstar)
		print "Tracing Random"
		obj_plot_random = trace_objective_function_CL("random", data, lambda_, wstar)


		plt.figure(2)
		plt.plot(obj_plot_hard[10:])
		plt.plot(obj_plot_easy[10:])
		plt.plot(obj_plot_random[10:])
		plt.legend(['Hard Examples First', 'Easy Examples First', 'Normal/Random Examples'], loc='lower right')
		plt.ylabel("Objective Function Value")

		plt.show()	
	else:
		print "Wrong number of arguments"
main()