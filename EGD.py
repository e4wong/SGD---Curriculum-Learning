import numpy
from numpy import linalg, random as rn
import matplotlib.pyplot as plt
import sys
import random
import math
import copy
from library import *

exponents = [-2, -1, 0]
base = 10
stepsize_constant_var = 1


def find_lambda(training_set, validation_set, stepsize_constant):
    # configurations of lambda
    global exponents
    global base 
    
    #defaults to compare to
    best_mistakes = len(validation_set)
    best_lambda = 1
    for exp in exponents:
        curr_lambda = float(math.pow(base, exp))
        (w, errors) = EGD(training_set, stepsize_constant, curr_lambda, False, validation_set) 
        mistakes = count_errors(w, validation_set)
        if mistakes < best_mistakes:
            best_mistakes = mistakes
            best_lambda = curr_lambda
    return best_lambda

def run_EGD(training_set, validation_set, stepsize_constant, plot):
    best_lambda = find_lambda(training_set,validation_set, stepsize_constant)
    return EGD(training_set, stepsize_constant, best_lambda, plot, validation_set)

def get_w(w_plus, w_minus):
    w_final = [plus - minus for (plus, minus) in zip(w_plus, w_minus)]
    return w_final

def EGD(training_set, stepsize_constant, lambda_, error_log, validation_set):
    dimensions = len(training_set[0][0])
    w_plus = [1.0/(2.0 * dimensions)] * dimensions
    w_minus = [1.0/(2.0 * dimensions)] * dimensions 
    errors = []
    errors.append(total_error(get_w(w_plus, w_minus), lambda_, validation_set))
    for i in range(0, len(training_set)):
        # I think Shuang is using stepsize = 1 currently
        stepsize = stepsize_constant
        datum = training_set[i]
        gradient = derivative(datum, get_w(w_plus,w_minus))
        r_t = [numpy.exp(-1.0 * stepsize * w_i) for w_i in gradient]

        next_w_plus = [prev_w_plus_i * r_i for (prev_w_plus_i, r_i) in zip(w_plus, r_t)]
        next_w_minus = [prev_w_minus_i * 1.0/r_i for (prev_w_minus_i, r_i) in zip(w_minus, r_t)]
        
        total_w_plus_minus = sum(next_w_plus) + sum(next_w_minus)

        next_w_plus = [val / total_w_plus_minus for val in next_w_plus]
        next_w_minus = [val / total_w_plus_minus for val in next_w_minus]

        w_plus = next_w_plus
        w_minus = next_w_minus
        errors.append(total_error(get_w(w_plus, w_minus), lambda_, validation_set))
    w_final = get_w(w_plus, w_minus)
    return (w_final, errors)

def main():
    if len(sys.argv) < 2:
        print "Wrong way to use me!"
        return

    elif len(sys.argv) == 2:
        filename = sys.argv[1]    
        (wstar, data) = load_data(filename)
        random.shuffle(data)

        training_set = data[len(data)/2 :]
        validation_set = data[: len(data)/2]
        # (result, errors) = run_EGD(training_set, validation_set, stepsize_constant_var, True)
        (result, errors) = EGD(training_set, .2, .001, True, validation_set)
        print "Validation Set Error Rate: " + str(calc_error_rate(result, validation_set))
        output_final_w(result)
        plt.plot(errors)
        plt.ylabel('Objective Function')
        plt.show()

    else:
        print "Wrong number of arguments"
main()