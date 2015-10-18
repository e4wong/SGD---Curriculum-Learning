import sys
import random
import math

def load_data(filename):
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

def dotproduct(v1, v2):
 	return sum((a*b) for a, b in zip(v1, v2))

def length(v):
	return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
	return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def cos(val):
	return math.cos(val)

def random_noise(prob_mislabel, original_data):
	noisy_data = []
	for (features,label) in original_data:
		sign = 1
		rand = random.uniform(0.0, 100.0)
		if rand <= prob_mislabel:
			sign = -1
		noisy_data.append((features, label*sign))
	return noisy_data

k = 10.0
def noise(labeled_data):
	noisy_data = []
	for ((features,label), hardness) in labeled_data:
		sign = 1
		rand = random.uniform(0.0, 1.0)
		prob = (1.0 - hardness)/k
		if rand <= prob:
			sign = -1
		noisy_data.append((features, label*sign))
	return noisy_data


def label_data_hardness(data, w_star):
	# sanity check, should always be hard or easy
	data_hardness = []
	for (features, label) in data:
		cos_val = cos(angle(w_star,features))
		data_hardness.append(((features,label), abs(cos_val)))
	return data_hardness

def print_dataset(wstar, labeled_data):
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

def main():
	if len(sys.argv) < 2:
		print "Please enter a file and percentage of points to "
		return
	elif len(sys.argv) == 2:
		filename = sys.argv[1]
		(wstar, data) = load_data(filename)
		labeled_data = label_data_hardness(data, wstar)
		noisy_data = noise(labeled_data)
		print_dataset(wstar, noisy_data)
	elif len(sys.argv) == 3:
		filename = sys.argv[1]	
		(wstar, data) = load_data(filename)
		prob = float(sys.argv[2])
		if prob <= 0.0 or prob >= 100.0:
			print prob, "is out of range 0.0 - 100.0"
		noisy_data = random_noise(prob, data)
		print_dataset(wstar, noisy_data)

main()