from sklearn.decomposition import PCA
import sys

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

def pca_transform_data(wstar, data, dimensions):
	X = [features for (features,label) in data]
	y = [label for (features,label) in data]
	pca = PCA(n_components=dimensions)
	pca.fit(X)
	transformed_data = []
	for i in range(0, len(data)):
		X_transform = pca.transform(X[i])
		transformed_data.append((X_transform[0],y[i]))
	wstar_transformed = pca.transform(wstar)

	return (wstar_transformed[0], transformed_data)

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

def append_constant_feature(wstar, data):
	final_wstar = [1.0]
	for i in range(0, len(wstar)):
		final_wstar.append(wstar[i])
	final_data = []
	for (features,label) in data:
		final_f = [1.0]
		for i in range(0, len(features)):
			final_f.append(features[i])
		final_data.append((final_f,label))

	return(final_wstar, final_data)

def main():
	if len(sys.argv) < 3:
		print "Please enter a file and dimensions to reduce to"
		return

	filename = sys.argv[1]
	dimensions = int(sys.argv[2])
	(wstar, data) = load_data(filename)
	(wstar_transformed, transformed_data) = pca_transform_data(wstar, data, dimensions)
	(w_final, final_data) = append_constant_feature(wstar_transformed, transformed_data)
	print_dataset(w_final, final_data)

main()