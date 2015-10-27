import numpy
from numpy import linalg, random as rn
import matplotlib.pyplot as plt
import sys
import random
import time
import math
import scipy.optimize
import copy
from sklearn.linear_model import SGDClassifier


def f(theta, X, y, lam):
  theta = numpy.matrix(theta).T
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  diffSq = diff.T*diff
  diffSqReg = diffSq / len(X) + lam*(theta.T*theta)
  print "offset =", diffSqReg.flatten().tolist()
  return diffSqReg.flatten().tolist()[0]

# Derivative
def fprime(theta, X, y, lam):
  theta = numpy.matrix(theta).T
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  res = 2*X.T*diff / len(X) + 2*lam*theta
  return numpy.array(res.flatten().tolist()[0])

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

exponents = [-2, -1, 0]
base = 10

def gradient_descent(training_set, validation_set):
  global exponents
  global base
  best_result = None
  best_error = 1.0
  for exp in exponents:
    curr_lambda = float(math.pow(base, exp))
    
    X = [features for (features,label) in training_set]
    y = [label for (features,label) in training_set]
    init = rn.normal(size=(1, len(training_set[0][0])))[0]
    (result, val, _) = scipy.optimize.fmin_l_bfgs_b(f, init, fprime, args = (X, y, curr_lambda))
    curr_error = calc_error_rate(result, validation_set)
    if curr_error < best_error:
      best_result = result
      best_error = curr_error
  return best_result

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

    result = gradient_descent(training_set, validation_set)

    print "Validation Set Error Rate: " + str(calc_error_rate(result, validation_set))
    output_final_w(result)

main()