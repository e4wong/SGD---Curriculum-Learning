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

dimensions_zero = 1.0/3.0 # fraction of dimensions to want to zero out
hard_example_fraction = 1.0/5.0 # fraction of datapoints to apply zeroout to

def generate_dimensions_to_zero_out(dimensions_number):
	dimensions_to_zero_out = []
	for i in range(0, dimensions_number):
		rand = random.uniform(0.0, 1.0)
		if rand <= dimensions_zero:
			dimensions_to_zero_out.append(i)
	return dimensions_to_zero_out	

def generate_hard_examples(data, zero_dimensions):
	labeled_data = []
	for (features,label) in data:
		rand = random.uniform(0.0, 1.0)
		if rand <= hard_example_fraction:
			modified_features = []
			for i in range(0, len(features)):
				if i in zero_dimensions:
					modified_features.append(0.0)
				else:
					modified_features.append(features[i])
			labeled_data.append(((modified_features,label), 1))
		else:
			labeled_data.append(((features,label), 0))
	return labeled_data

def print_dataset(wstar, labeled_data):
	wstarconfig =  "//w* = " + str(wstar)
	wstarconfig = wstarconfig.replace("\n", "")
	wstarconfig = wstarconfig.replace(",", "")
	print wstarconfig
	for ((data,label),hard) in labeled_data:
		string = ""
		for dimension in data:
			string += str(dimension) + " "
		string += str(label)
		print string

default_output_file = "hard_examples."
def output_hard_examples(filename, labeled_data):
	f = open(default_output_file + filename,'w')
	for ((data,label),hard) in labeled_data:
		if not(hard == 1):
			continue
		string = ""
		for dimension in data:
			string += str(dimension) + " "
		string += str(label) + "\n"
		f.write(string)
		
	f.flush()
	f.close()



def main():
	if len(sys.argv) < 2:
		print "Please enter a file"
		return
	elif len(sys.argv) == 2:
		filename = sys.argv[1]
		(wstar, data) = load_data(filename)
		dimensions_number = len(data[0][0])
		zero_dimensions = generate_dimensions_to_zero_out(dimensions_number)
		modified_data = generate_hard_examples(data, zero_dimensions)
		print_dataset(wstar, modified_data)
		output_hard_examples(filename, modified_data)
main()
