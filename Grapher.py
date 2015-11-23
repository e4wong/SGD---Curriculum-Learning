import sys
import matplotlib.pyplot as plt
import math

def get_array(s):
	s = s[s.index('[') + 1 : len(s) - 2]
	tokens = s.split()
	a = []
	for token in tokens:
		a.append(float(token))
	return a

def load_data(filename):
	f = open(filename, "r")
	data = []
	for i in range(0,5):
		name = next(f).replace("\n", "")

		error_rate = get_array(next(f))
		final_obj_val = get_array(next(f))
		trace_obj_val = get_array(next(f))
		data.append((name, (error_rate, final_obj_val, trace_obj_val)))
	return data

def calc_mean(data):
	total = 0.0
	for value in data:
		total += value
	return total / len(data)

def calc_std_dev(data, mean):
	total = 0.0
	for value in data:
		total += (value - mean) ** 2
	total = total / len(data)
	return math.sqrt(total)


def main():
	if len(sys.argv) < 2:
		print "Wrong way to use me!"
		return
	elif len(sys.argv) == 2:
		filename = sys.argv[1]	
		data = load_data(filename)
		# error_rate
		f_0 = plt.figure(0)
		f_1 = plt.figure(1)
		f_2 = plt.figure(2)
		names = []
		for (name, (error_rate, final_obj_val, trace_obj_val)) in data:
			names.append(name)
			f_0 = plt.figure(0)
			plt.plot(error_rate)
			mean_er = calc_mean(error_rate)
			print name, "Mean Error Rate:", mean_er
			print name, "Error Rate Standard Deviation:", calc_std_dev(error_rate, mean_er)

			f_1 = plt.figure(1)
			plt.plot(final_obj_val)
			mean_final_obj_val = calc_mean(final_obj_val)
			print name, "Mean Final Objective Function Value:", mean_final_obj_val
			print name, "Final Objective Function Value Standard Deviation:", calc_std_dev(final_obj_val, mean_final_obj_val)

			f_2 = plt.figure(2)
			plt.plot(trace_obj_val[5:])
			print ""

		f_0 = plt.figure(0)
		plt.legend(names, loc='upper left')
		plt.ylabel('Error Rate')
		f_1 = plt.figure(1)
		plt.legend(names, loc='upper right')
		plt.ylabel("Final Objective Function Value")
		f_2 = plt.figure(2)
		plt.legend(names, loc='upper right')
		plt.ylabel("Objective Function Value")
		plt.show()	


main()