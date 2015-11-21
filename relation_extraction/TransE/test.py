import random
def rand_max(x):
	res = ((random.randint(0,100))*(random.randint(0,100))) % x
	while res < 0:
		print res
		res += x
	return res

rand_max(13)
