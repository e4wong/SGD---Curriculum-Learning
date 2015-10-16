import sys
import random

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

def random_noise(prob_mislabel, original_data):
	noisy_data = []
	for (features,label) in original_data:
		sign = 1
		rand = random.uniform(0.0, 100.0)
		if rand <= prob_mislabel:
			sign = -1
		noisy_data.append((features, label*sign))
	return noisy_data

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
	if len(sys.argv) < 3:
		print "Please enter a file and percentage of points to "
		return

	filename = sys.argv[1]	
	(wstar, data) = load_data(filename)
	prob = float(sys.argv[2])
	if prob <= 0.0 or prob >= 100.0:
		print prob, "is out of range 0.0 - 100.0"
	noisy_data = random_noise(prob, data)
	print_dataset(wstar, noisy_data)

main()