import sys

workclass = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
education = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
marital_status = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
occupation = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
relationship = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
race = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
sex = ["Female", "Male"]
native_country = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]

def load_data(filename):
	f = open(filename, "r")
	ds = []
	i = 0
	for line in f:
		i += 1
		features = []
		sign = 0
		tokens = line.replace(',',' ').split()
		flag = False
		for token in tokens:
			if token == "?":
				flag = True
				break


		if flag:
			continue

		i += 1
		if i % 3 == 0 and tokens[len(tokens) - 1] == "<=50K" :
			continue

		# age
		features.append(float(tokens[0]))
		#workclass
		found = False
		for i in range(0, len(workclass)):
			if tokens[1] == workclass[i]:
				features.append(1.0)
				found = True
			else:
				features.append(0.0)

		if not(found):
			print "uhoh"
			break
		# fnlwage
		features.append(float(tokens[2]))	
		#education
		found = False
		for i in range(0, len(education)):
			if tokens[3] == education[i]:
				features.append(1.0)
				found = True
			else:
				features.append(0.0)


		if not(found):
			print tokens
			print "uhoh"
			break
		# education num
		features.append(float(tokens[4]))
		# marital status
		found = False
		for i in range(0, len(marital_status)):
			if tokens[5] == marital_status[i]:
				found = True
				features.append(1.0)
			else:
				features.append(0.0)


		if not(found):
			print tokens
			print "uhoh"
			break
		# occupation
		found = False

		for i in range(0, len(occupation)):
			if tokens[6] == occupation[i]:
				found = True
				features.append(1.0)
			else:
				features.append(0.0)

		if not(found):
			print tokens
			print "uhoh"
			break

		# relationship
		found = False

		for i in range(0, len(relationship)):
			if tokens[7] == relationship[i]:
				found = True
				features.append(1.0)
			else:
				features.append(0.0)

		if not(found):
			print tokens
			print "uhoh"
			break

		# race
		found = False

		for i in range(0, len(race)):
			if tokens[8] == race[i]:
				found = True
				features.append(1.0)
			else:
				features.append(0.0)

		if not(found):
			print tokens
			print "uhoh"
			break
		#sex 
		found = False

		for i in range(0, len(sex)):
			if tokens[9] == sex[i]:
				found = True
				features.append(1.0)
			else:
				features.append(0.0)

		if not(found):
			print tokens
			print "uhoh"
			break

		# capital gain
		features.append(float(tokens[10]))
		# capital loss
		features.append(float(tokens[11]))
		# hours per week
		features.append(float(tokens[12]))
		#native country
		found = False

		for i in range(0, len(native_country)):
			if tokens[13] == native_country[i]:
				found = True
				features.append(1.0)
			else:
				features.append(0.0)

		if not(found):
			print tokens
			print "CRAP"
			break

		features.append(1.0)
		#label
		if tokens[14] == "<=50K":
			ds.append((features, -1))
		elif tokens[14] == ">50K":
			ds.append((features, 1))
		
	return ds

def normalize(data):
	features_to_normalize = [0,9,26,61,62,63]
	for feature in features_to_normalize:
		minimal = float("inf")
		maximal = 0.0
		for (datum, label) in data:
			if datum[feature] < minimal:
				minimal = datum[feature]
			elif datum[feature] > maximal:
				maximal = datum[feature]
		for (datum, label) in data:
			datum[feature] = (datum[feature] - minimal) / (maximal - minimal)

def print_dataset(labeled_data):
    wstarconfig =  "//w* = " + "[]"
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
		print "Wrong way to use me!"
		return
	elif len(sys.argv) == 2:
		filename = sys.argv[1]	
		data = load_data(filename)
		normalize(data)
		print_dataset(data)
		


main()
