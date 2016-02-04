import numpy
from numpy import linalg, random as rn
import matplotlib.pyplot as plt
import sys
import random
import math
import copy
from library import *

exponents = [-8, -7, -6, -5, -4, -3, -2, -1, 0]
base = 10
stepsize_constant_var = .2


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

def output_final_data(fn, data):
    f = open(fn + ".OUTPUTDATA", 'w')
    for (label, (error_rate, final_objective_func_val, avg_objective_func_trace, error_rate_trace)) in data:
        f.write(label + "\n")
        f.write(str(error_rate).replace(",", "") + "\n")
        f.write(str(final_objective_func_val).replace(",", "") + "\n")
        f.write(str(avg_objective_func_trace).replace(",", "") + "\n")
        f.write(str(error_rate_trace).replace(",", "") + "\n")

def run_EGD(training_set, validation_set, stepsize_constant, plot):
    best_lambda = find_lambda(training_set,validation_set, stepsize_constant)
    return EGD(training_set, stepsize_constant, best_lambda, plot, validation_set)

def get_w(w_plus, w_minus):
    w_final = [plus - minus for (plus, minus) in zip(w_plus, w_minus)]
    return w_final

def find_lambda(training_set, validation_set, stepsize_constant):
    global exponents
    global base
    # no regularization for EGD anymore
    return 0.0
    best_mistakes = len(validation_set)
    best_lambda = 1.0
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
    print "Best Lambda:", best_lambda
    return EGD(training_set, stepsize_constant, best_lambda, plot, validation_set)

def EGD(training_set, stepsize_constant, lambda_, error_log, validation_set):
    dimensions = len(training_set[0][0])
    w_plus = [1.0/(2.0 * dimensions)] * dimensions
    w_minus = [1.0/(2.0 * dimensions)] * dimensions

    mod_by = 5

    errors = []
    error_rate_trace = []
    error_rate_trace.append(calc_error_rate(get_w(w_plus, w_minus), validation_set))
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
        if error_log and (i % mod_by) == 0:
            #Bottle neck right now
            errors.append(total_error(get_w(w_plus, w_minus), lambda_, validation_set))
            error_rate_trace.append(calc_error_rate(get_w(w_plus, w_minus), validation_set))
    w_final = get_w(w_plus, w_minus)
    return (w_final, (errors, error_rate_trace))

def magnitude_irrelevant_features(number_irrelevant_features, features):
    total = 0.0
    for i in range(len(features) - number_irrelevant_features, len(features)):
        total = total + abs(features[i])
    return total


def sort_data(sort_by, data, number_irrelevant_features):
    if not(sort_by == "hard") and not(sort_by == "easy") and not(sort_by == "random") and not(sort_by == "hardhalf") and not(sort_by == "easyhalf"):
        print "Something went wrong!!!!!!!"
        return [] 

    if sort_by == "random":
        return data

    data_hardness = []
    for (features, label) in data:
        data_hardness.append(((features,label), magnitude_irrelevant_features(number_irrelevant_features, features)))
    
    data_hardness.sort(key=lambda tup: tup[1])

    # Hard means that hard examples come first, which means the lower abs(cos_val) are first in the list
    # Easy means that easy examples come first, which means the higher abs(cos_val) are first in the list
    if sort_by == "hard" or sort_by == "hardhalf":
        data_hardness.reverse()


    train = [data for (data, hardness) in data_hardness]    

    if sort_by == "easyhalf" or sort_by == "hardhalf":
        return train[ :len(train)/2]
    return train




def curriculum_learning(sort_by, times_to_run, original_data, number_irrelevant_features, lambda_ = None):
    global stepsize_constant_var
    error_rate = []
    final_objective_func_val = []
    avg_trace_objective_func_val = []
    avg_error_rate_trace = []
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
        training_set = sort_data(sort_by, training_set, number_irrelevant_features)
        (w, (errors, error_rate_trace)) = EGD(training_set, stepsize_constant_var, lambda_, True, validation_set)
        if len(avg_error_rate_trace) == 0:
            avg_error_rate_trace = error_rate_trace
        elif len(avg_error_rate_trace) == len(error_rate_trace):
            tmp = [avg_error_rate_trace[i] + error_rate_trace[i] for i in range(0, len(avg_error_rate_trace))]
            avg_error_rate_trace = tmp
        else:
            print "Visit this problem"

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
    for i in range(0, len(avg_error_rate_trace)):
        avg_error_rate_trace[i] = avg_error_rate_trace[i]/times_to_run
    for i in range(0, len(avg_trace_objective_func_val)):
        avg_trace_objective_func_val[i] = avg_trace_objective_func_val[i]/times_to_run

    return (error_rate, final_objective_func_val, avg_trace_objective_func_val, avg_error_rate_trace)

def get_number_irrelevant_features(training_sample):
    count = 0
    for i in range(len(training_sample) - 1, -1, -1):
        if not(training_sample[i] == 0.0):
            break
        count = count + 1
    return count


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
        (result, (errors, error_rate)) = run_EGD(training_set, validation_set, stepsize_constant_var, True)
        print "Validation Set Error Rate: " + str(calc_error_rate(result, validation_set))
        output_final_w(result)
        f_0 = plt.figure(0)
        f_0.canvas.set_window_title(filename)
        plt.plot(errors)
        plt.ylabel('Objective Function')

        f_1 = plt.figure(1)
        f_1.canvas.set_window_title(filename)
        plt.plot(error_rate)
        plt.ylabel('Error Rate')
        plt.show()

    elif len(sys.argv) == 3:
        filename = sys.argv[1]    
        (wstar, data) = load_data(filename)
        random.shuffle(data)
        num_runs = int(sys.argv[2])     
        print "Curriculum learning, Number of iterations for both hard and easy to run:", num_runs

        random.shuffle(data)
        training_set = data[len(data)/2 : ]
        validation_set = data[ : len(data)/2]
        lambda_ = find_lambda(training_set,validation_set, stepsize_constant_var)
        number_irrelevant_features = get_number_irrelevant_features(wstar)
        print "Lambda:", lambda_
        print "Number of Irrelevant Features:", get_number_irrelevant_features(wstar)
        print "Running Hard->Easy"
        (hard_error_rate, hard_objective_func_val, obj_plot_hard, hard_err_trace) = curriculum_learning("hard", num_runs, data, number_irrelevant_features, lambda_)
        print "Running Easy->Hard"
        (easy_error_rate, easy_objective_func_val, obj_plot_easy, easy_err_trace) = curriculum_learning("easy", num_runs, data, number_irrelevant_features, lambda_)
        print "Running Random"
        (random_error_rate, random_objective_func_val, obj_plot_random, random_err_trace) = curriculum_learning("random", num_runs, data, number_irrelevant_features, lambda_)
        print "Running Hard Half"
        (hh_error_rate, hh_objective_func_val, obj_plot_hh, hh_err_trace) = curriculum_learning("hardhalf", num_runs, data, number_irrelevant_features, lambda_)
        print "Running Easy Half"
        (eh_error_rate, eh_objective_func_val, obj_plot_eh, eh_err_trace) = curriculum_learning("easyhalf", num_runs, data, number_irrelevant_features, lambda_)

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
        plt.ylabel("Objective Function Value Trace")



        plt.figure(3)
        f_2 = plt.figure(3)
        f_2.canvas.set_window_title(filename)
        plt.plot(hard_err_trace)
        plt.plot(easy_err_trace)
        plt.plot(random_err_trace)
        plt.plot(hh_err_trace)
        plt.plot(eh_err_trace)
        plt.legend(['Hard Examples First', 'Easy Examples First', 'Normal/Random Examples', 'Hard Half', 'Easy Half'], loc='upper right')
        plt.ylabel("Error Rate Trace")

        plt.show()  
        print "Outputting values"
        final_data = []
        final_data.append(("Hard Examples First",(hard_error_rate, hard_objective_func_val, obj_plot_hard, hard_err_trace)))
        final_data.append(("Easy Examples First",(easy_error_rate, easy_objective_func_val, obj_plot_easy, easy_err_trace)))
        final_data.append(("Random Examples First",(random_error_rate, random_objective_func_val, obj_plot_random, random_err_trace)))
        final_data.append(("Hard Half",(hh_error_rate, hh_objective_func_val, obj_plot_hh, hh_err_trace)))
        final_data.append(("Easy Half",(eh_error_rate, eh_objective_func_val, obj_plot_eh, eh_err_trace)))
        output_final_data(filename + "_lambda:" + str(lambda_), final_data)
 

    else:
        print "Wrong number of arguments"
main()