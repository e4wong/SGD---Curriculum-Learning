import numpy
from numpy import linalg, random as rn
import matplotlib.pyplot as plt
import sys
import random
import math
import copy
from sklearn.linear_model import SGDClassifier
from library import *

exponents = [-3, -2, -1, 0]
base = 10

stepsize_constant_var = 1


def load_hard_examples(filename):
	f = open(filename, "r")
	print "Reading hard examples from file " + filename
	
	hard_data = []
	for line in f:
		features = []
		sign = 0
		tokens = line.split()
		for i in range(0, len(tokens) - 1):
			features.append(float(tokens[i]))
		sign = int(tokens[len(tokens) - 1])
		hard_data.append((features,sign))
	print "Done loading hard data"
	return hard_data	

def output_final_data(fn, data):
	f = open(fn + ".OUTPUTDATA", 'w')
	for (label, (error_rate, final_objective_func_val, avg_objective_func_trace)) in data:
		f.write(label + "\n")
		f.write(str(error_rate).replace(",", "") + "\n")
		f.write(str(final_objective_func_val).replace(",", "") + "\n")
		f.write(str(avg_objective_func_trace).replace(",", "") + "\n")


def total_error_matrix_optimize(w, lambda_, dataset):
	x = [features for (features, label) in dataset]
	y = [label for (features, label) in dataset]
	x = numpy.matrix(x)
	w_ = numpy.matrix(w)
	wTx = x * w_.T
	wTx = numpy.squeeze(numpy.asarray(wTx))
	e_neg_wTx_y = [numpy.log(1 + numpy.exp(-1 * wTx[i] * y[i])) for i in range(0, len(wTx))]
	return sum(e_neg_wTx_y)/len(e_neg_wTx_y) + (float(lambda_)/2.0) * linalg.norm(w) ** 2



def scale_w(lambda_, w):
	# if lambda = .001 , then 1/.001 causes overflow when we do e^ in derivative
	mag = float(1)/ float(lambda_)
	if linalg.norm(w) > mag:
		w = (1.0/linalg.norm(w)) * (1.0/float(lambda_))  * numpy.array(w)
	return w


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

def SGD(training_set, stepsize_constant, lambda_, error_log, validation_set, init_w=None):
	#print "Starting SGD algorithm with stepsize_constant: " + str(stepsize_constant) + " lambda: " + str(lambda_)
	init = rn.normal(size=(1, len(training_set[0][0])))[0]

	mod_by = 15
	'''
	if len(training_set) < 2000:
		mod_by = 2
	elif len(training_set) < 10000:
		mod_by = 10
	else:
		mod_by = 20
	'''
	if not(init_w is None):
		init = init_w
		print "Initialized w for SGD run"

	w = scale_w(lambda_, init)
	errors = []
	errors.append(total_error(w, lambda_, validation_set))
	for i in range(0, len(training_set)):
		datum = training_set[i]
		# Stepsize is current c / sqrtroot(t)
		stepsize = stepsize_fn(stepsize_constant, lambda_, i + 1)
		delta = derivative(datum, w) + lambda_ * numpy.array(w)
		delta = stepsize * numpy.array(delta)
		w = w - delta
		w = scale_w(lambda_, w)
		if error_log and (i % mod_by) == 0:
			#Bottle neck right now
			errors.append(total_error(w, lambda_, validation_set))
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
	if not(sort_by == "hard") and not(sort_by == "easy") and not(sort_by == "random") and not(sort_by == "hardhalf") and not(sort_by == "easyhalf"):
		print "Something went wrong!!!!!!!"
		return [] 

	if sort_by == "random":
		return data

	data_hardness = []
	for (features, label) in data:
		cos_val = cos(angle(w_star,features))
		data_hardness.append(((features,label), abs(cos_val)))

	data_hardness.sort(key=lambda tup: tup[1])

	# Hard means that hard examples come first, which means the lower abs(cos_val) are first in the list
	# Easy means that easy examples come first, which means the higher abs(cos_val) are first in the list
	if sort_by == "easy" or sort_by == "easyhalf":
		data_hardness.reverse()
	train = [data for (data, hardness) in data_hardness]	

	if sort_by == "easyhalf" or sort_by == "hardhalf":
		return train[ :len(train)/2]
	return train


def sort_data0(sort_by, data, hard_data):
	# sanity check, should always be hard or easy
	if not(sort_by == "hard") and not(sort_by == "easy") and not(sort_by == "random") and not(sort_by == "justeasy") and not(sort_by == "justhard"):
		print "Something went wrong!!!!!!!"
		return [] 

	if sort_by == "random":
		return data

	data_hardness = []
	if sort_by == "justeasy" or sort_by == "justhard":
		for datum in data:
			if datum in hard_data:
				if sort_by == "justeasy":
					continue
				data_hardness.append((datum, 1))
			else:
				if sort_by == "justhard":
					continue
				data_hardness.append((datum, 0))
	else:
		for datum in data:
			if datum in hard_data:
				data_hardness.append((datum, 1))
			else:
				data_hardness.append((datum, 0))
	
	data_hardness.sort(key=lambda tup: tup[1])
	if sort_by == "hard":
		data_hardness.reverse()
	train = [data for (data, hardness) in data_hardness]	
	return train

def curriculum_learning0(sort_by, times_to_run, original_data, hard_data, lambda_):
	global stepsize_constant_var
	error_rate = []
	final_objective_func_val = []
	avg_trace_objective_func_val = []
	for i in range(0, times_to_run):
		data = copy.deepcopy(original_data)
		random.shuffle(data)
		training_set = data[len(data)/2 : ]
		validation_set = data[ : len(data)/2]
		training_set = sort_data0(sort_by, training_set, hard_data)
		(w, errors) = SGD(training_set, stepsize_constant_var, lambda_, True, validation_set)
		if len(avg_trace_objective_func_val) == 0:
			avg_trace_objective_func_val = errors
		else:
			length = 0.0
			if len(avg_trace_objective_func_val) < len(errors):
				length = len(avg_trace_objective_func_val)
			else:
				length = len(errors)
			for i in range(0, length):
				if avg_trace_objective_func_val[i] == float("inf"):
					avg_trace_objective_func_val[i] = errors[i]
				elif errors[i] == float("inf"):
					continue
				else:
					avg_trace_objective_func_val[i] += errors[i]

		error_rate.append(calc_error_rate(w, validation_set))
		final_objective_func_val.append(total_error(w, lambda_, validation_set))
	for i in range(0, len(avg_trace_objective_func_val)):
		avg_trace_objective_func_val[i] = avg_trace_objective_func_val[i]/times_to_run
	return (error_rate, final_objective_func_val, avg_trace_objective_func_val)


def curriculum_learning(sort_by, times_to_run, original_data, w_star, lambda_ = None):
	global stepsize_constant_var
	error_rate = []
	final_objective_func_val = []
	avg_trace_objective_func_val = []
	if lambda_ is None:
		data = copy.deepcopy(original_data)
		random.shuffle(data)
		training_set = data[len(data)/2 : ]
		validation_set = data[ : len(data)/2]
		training_set = sort_data(sort_by, training_set, w_star)
		lambda_ = find_lambda(training_set,validation_set, stepsize_constant_var)
	print "Lambda for", sort_by, "is", lambda_

	for i in range(0, times_to_run):
		data = copy.deepcopy(original_data)
		random.shuffle(data)
		training_set = data[len(data)/2 : ]
		validation_set = data[ : len(data)/2]
		training_set = sort_data(sort_by, training_set, w_star)
		(w, errors) = SGD(training_set, stepsize_constant_var, lambda_, True, validation_set)
		if len(avg_trace_objective_func_val) == 0:
			avg_trace_objective_func_val = errors
		else:
			length = 0.0
			if len(avg_trace_objective_func_val) < len(errors):
				length = len(avg_trace_objective_func_val)
			else:
				length = len(errors)
			for i in range(0, length):
				if avg_trace_objective_func_val[i] == float("inf"):
					avg_trace_objective_func_val[i] = errors[i]
				elif errors[i] == float("inf"):
					continue
				else:
					avg_trace_objective_func_val[i] += errors[i]		
		error_rate.append(calc_error_rate(w, validation_set))
		final_objective_func_val.append(total_error(w, lambda_, validation_set))
	for i in range(0, len(avg_trace_objective_func_val)):
		avg_trace_objective_func_val[i] = avg_trace_objective_func_val[i]/times_to_run
	return (error_rate, final_objective_func_val, avg_trace_objective_func_val)


def trace_objective_function_CL0(sort_by, data, lambda_, hard_data):
	random.shuffle(data)
	training_set = data[len(data)/2 : ]
	validation_set = data[ : len(data)/2]
	training_set = sort_data0(sort_by, training_set, hard_data)
	(w, obj_plot) = SGD(training_set, stepsize_constant_var, lambda_, True, validation_set)
	return obj_plot


def trace_objective_function_CL(sort_by, data, lambda_, w_star):
	random.shuffle(data)
	training_set = data[len(data)/2 : ]
	validation_set = data[ : len(data)/2]
	training_set = sort_data(sort_by, training_set, w_star)
	(w, obj_plot) = SGD(training_set, stepsize_constant_var, lambda_, True, validation_set)
	return obj_plot

def main():
	if len(sys.argv) < 2:
		print "Wrong way to use me!"
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
		print "Lambda:", lambda_
		print "Running Hard->Easy"
		(hard_error_rate, hard_objective_func_val, obj_plot_hard) = curriculum_learning("hard", num_runs, data, wstar, lambda_)
		print "Running Easy->Hard"
		(easy_error_rate, easy_objective_func_val, obj_plot_easy) = curriculum_learning("easy", num_runs, data, wstar, lambda_)
		print "Running Random"
		(random_error_rate, random_objective_func_val, obj_plot_random) = curriculum_learning("random", num_runs, data, wstar, lambda_)
		print "Running Hard Half"
		(hh_error_rate, hh_objective_func_val, obj_plot_hh) = curriculum_learning("hardhalf", num_runs, data, wstar, lambda_)
		print "Running Easy Half"
		(eh_error_rate, eh_objective_func_val, obj_plot_eh) = curriculum_learning("easyhalf", num_runs, data, wstar, lambda_)

		print "Error Rate of w*:", calc_error_rate(wstar, data)

		f_0 = plt.figure(0)
		f_0.canvas.set_window_title(filename)
		plt.plot(hard_error_rate)
	 	plt.plot(easy_error_rate)
	 	plt.plot(random_error_rate)
	 	plt.plot(hh_error_rate)
	 	plt.plot(eh_error_rate)
		plt.legend(['Hard Examples First', 'Easy Examples First', 'Normal/Random Examples', 'Hard Half', 'Easy Half'], loc='upper left')
		plt.ylabel('Error Rate')

		f_1 = plt.figure(1)
		f_1.canvas.set_window_title(filename)
		plt.plot(hard_objective_func_val)
		plt.plot(easy_objective_func_val)
		plt.plot(random_objective_func_val)
		plt.plot(hh_objective_func_val)
		plt.plot(eh_objective_func_val) 
		plt.legend(['Hard Examples First', 'Easy Examples First', 'Normal/Random Examples', 'Hard Half', 'Easy Half'], loc='lower right')
		plt.ylabel("Final Objective Function Value")
	


		plt.figure(2)
		f_2 = plt.figure(2)
		f_2.canvas.set_window_title(filename)
		plt.plot(obj_plot_hard[10:])
		plt.plot(obj_plot_easy[10:])
		plt.plot(obj_plot_random[10:])
		plt.plot(obj_plot_hh[10:])
		plt.plot(obj_plot_eh[10:])
		plt.legend(['Hard Examples First', 'Easy Examples First', 'Normal/Random Examples', 'Hard Half', 'Easy Half'], loc='upper right')
		plt.ylabel("Objective Function Value")

		plt.show()	
		print "Outputting values"
		final_data = []
		final_data.append(("Hard Examples First",(hard_error_rate, hard_objective_func_val, obj_plot_hard)))
		final_data.append(("Easy Examples First",(easy_error_rate, easy_objective_func_val, obj_plot_easy)))
		final_data.append(("Random Examples First",(random_error_rate, random_objective_func_val, obj_plot_random)))
		final_data.append(("Hard Half",(hh_error_rate, hh_objective_func_val, obj_plot_hh)))
		final_data.append(("Easy Half",(eh_error_rate, eh_objective_func_val, obj_plot_eh)))
		output_final_data(filename + "_lambda:" + str(lambda_), final_data)
	
	elif len(sys.argv) == 4:
		filename = sys.argv[1]
		hard_filename = sys.argv[2]	
		num_runs = int(sys.argv[3])		
		(wstar, data) = load_data(filename)
		hard_data = load_hard_examples(hard_filename)

		print "Curriculum learning for zero'ed out data, Number of iterations for both hard and easy to run:", num_runs 
		
		random.shuffle(data)
		training_set = data[len(data)/2 : ]
		validation_set = data[ : len(data)/2]
	
		lambda_ = find_lambda(training_set,validation_set, stepsize_constant_var)
		print "Lambda:", lambda_
		print "Running Hard->Easy"
		(hard_error_rate, hard_objective_func_val, obj_plot_hard) = curriculum_learning0("hard", num_runs, data, hard_data, lambda_)
		print "Running Easy->Hard"
		(easy_error_rate, easy_objective_func_val, obj_plot_easy) = curriculum_learning0("easy", num_runs, data, hard_data, lambda_)
		print "Running Random"
		(random_error_rate, random_objective_func_val, obj_plot_random) = curriculum_learning0("random", num_runs, data, hard_data, lambda_)
		print "Running Just Hard"
		(jh_error_rate, jh_objective_func_val, obj_plot_jh) = curriculum_learning0("justhard", num_runs, data, hard_data, lambda_)
		print "Running Just Easy"
		(je_error_rate, je_objective_func_val, obj_plot_je) = curriculum_learning0("justeasy", num_runs, data, hard_data, lambda_)


		f_0 = plt.figure(0)
		f_0.canvas.set_window_title(filename)
		plt.plot(hard_error_rate)
	 	plt.plot(easy_error_rate)
	 	plt.plot(random_error_rate)
	 	plt.plot(jh_error_rate)
	 	plt.plot(je_error_rate)
		plt.legend(['Hard Examples First', 'Easy Examples First', 'Normal/Random Examples', 'Just Hard', 'Just Easy'], loc='upper left')
		plt.ylabel('Error Rate')

		f_1 = plt.figure(1)
		f_1.canvas.set_window_title(filename)
		plt.plot(hard_objective_func_val)
		plt.plot(easy_objective_func_val)
		plt.plot(random_objective_func_val)
		plt.plot(jh_objective_func_val)
		plt.plot(je_objective_func_val)

		plt.legend(['Hard Examples First', 'Easy Examples First', 'Normal/Random Examples', 'Just Hard', 'Just Easy'], loc='upper right')
		plt.ylabel("Final Objective Function Value")


		plt.figure(2)
		f_2 = plt.figure(2)
		f_2.canvas.set_window_title(filename)
		plt.plot(obj_plot_hard[10:])
		plt.plot(obj_plot_easy[10:])
		plt.plot(obj_plot_random[10:])
		plt.plot(obj_plot_jh[10:])
		plt.plot(obj_plot_je[10:])
		plt.legend(['Hard Examples First', 'Easy Examples First', 'Normal/Random Examples', 'Just Hard', 'Just Easy'], loc='upper right')
		plt.ylabel("Objective Function Value")

		plt.show()	

		print "Outputting values"
		final_data = []
		final_data.append(("Hard Examples First",(hard_error_rate, hard_objective_func_val, obj_plot_hard)))
		final_data.append(("Easy Examples First",(easy_error_rate, easy_objective_func_val, obj_plot_easy)))
		final_data.append(("Random Examples First",(random_error_rate, random_objective_func_val, obj_plot_random)))
		final_data.append(("Just Hard",(jh_error_rate, jh_objective_func_val, obj_plot_jh)))
		final_data.append(("Just Easy",(je_error_rate, je_objective_func_val, obj_plot_je)))
		output_final_data(filename + "_lambda:" + str(lambda_), final_data)
	
	else:
		print "Wrong number of arguments"
main()