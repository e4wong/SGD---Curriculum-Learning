import sys
import copy
import random
from library import *

def generate_irrelevant_features(w_star, data, number_irrelevant_features):
    new_w_star = w_star + [0.0] * number_irrelevant_features
    new_data = []
    for (features, label) in data:
        ir_features = []
        for i in range(0, number_irrelevant_features):
            ir_features.append(random.uniform(-1.0,1.0))

        scaling_factor = random.uniform(0.5,1.5)
        ir_features = [feature * scaling_factor for feature in ir_features]
        new_data.append((features + ir_features, label))
    return (new_w_star, new_data)

NUMBER_CORRELATED_FEATURES = 5
def generate_irrelevant_correlated_features(w_star, data, number_irrelevant_features):
    new_w_star = w_star + [0.0] * number_irrelevant_features
    new_data = []
    for (features, label) in data:
        ir_features = []

        rand = random.uniform(-1.0, 1.0)
        correlated_features = []
        if rand < 0:
            correlated_features = [-1.0] * NUMBER_CORRELATED_FEATURES
        else:
            correlated_features = [1.0] * NUMBER_CORRELATED_FEATURES
        ir_features += correlated_features

        while len(ir_features) <number_irrelevant_features:
            ir_features.append(random.uniform(-1.0,1.0))

        scaling_factor = random.uniform(0.5,1.5)
        ir_features = [feature * scaling_factor for feature in ir_features]
        new_data.append((features + ir_features, label))
    return (new_w_star, new_data)

def main():
    if not(len(sys.argv) == 3):
        print "Please enter a file and # of irrelevant features to add"
        return
    elif len(sys.argv) == 3:
        filename = sys.argv[1]
        number_irrelevant_features = int(sys.argv[2])
        (w_star, data) = load_data_no_print(filename)
        (new_w_star, new_data) = generate_irrelevant_correlated_features(w_star, data, number_irrelevant_features)
        print_dataset_general(new_w_star, new_data)
main()