import numpy
from numpy import linalg, newaxis, random
import sys


def gen_rand_vecs(dims, number):
    vecs = random.normal(size=(number,dims))

    for i in range(0, len(vecs)):
        mags = linalg.norm(vecs[i])
        vecs[i] = vecs[i]/mags

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
    print "//w* = " + str(wstar)
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