import sys
import numpy
import random

def gen_rand_vecs(dims, number):
    vecs = []
    for i in range(0, number):
        vec = []
        for j in range(0, dims):
            vec.append(random.uniform(-1,1))
        vecs.append(vec)

    return vecs


def label_data(data):
    #Assume 1st sample is w*
    wstar = data[0]
    labeled_data = []
    for i in range(1, len(data)):
        datum = data[i]
        sign = 0
        if numpy.dot(datum,wstar) > 0:
            sign = 1
        else:
            sign = -1
        labeled_data.append((datum,sign)) 
    return labeled_data

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
    if len(sys.argv) != 3:
        print "Please enter dimensions first followed by number of samples"
        return
    dimensions = int(sys.argv[1])
    num_samples = int(sys.argv[2])
    # Generate an additional sample that will be our w*
    data = gen_rand_vecs(dimensions, num_samples + 1)
    labeled = label_data(data)
    print_dataset(data[0], labeled)


main()