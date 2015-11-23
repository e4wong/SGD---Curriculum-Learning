import sys
import matplotlib.pyplot as plt


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

			f_1 = plt.figure(1)
			plt.plot(final_obj_val)


			f_2 = plt.figure(2)
			plt.plot(trace_obj_val)

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