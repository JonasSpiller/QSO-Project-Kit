# LIBRARIES
import numpy as np
import time
from numpy.random import randint
from numpy.random import rand
from operator import itemgetter



## SET SEED 
np.random.seed(100)

# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		# store
		decoded.append(value)
	return decoded


# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] > scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]


# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]


# genetic algorithm as maximizer
def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut, X_train, y_train, X_valid, y_valid, strategy):
    
	start = time.time()
    
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
	# keep track of best solution
	best, accuracy_scores = pop[0], itemgetter('accuracy_valid_test', 'accuracy_train')(objective(decode(bounds, n_bits, pop[0]), X_train, y_train, X_valid, y_valid, strategy))#['accuracy_valid_test']
	# enumerate generations
	best_eval = accuracy_scores[0]
	temp_acc_train = accuracy_scores[1]
	gen = 0 
	s = 0 
	temp_decoded_params=[]
	best_accuracies_train=[]
	best_accuracies_valid=[]
	track_generation=[]
	track_hyperparams=[]
	for _ in range(n_iter):
		# decode population
		decoded = [decode(bounds, n_bits, p) for p in pop]
		# evaluate all candidates in the population
        
		accuracies_scores = [itemgetter('accuracy_valid_test', 'accuracy_train')(objective(d, X_train, y_train, X_valid, y_valid, strategy)) for d in decoded] # objective(d, X_train, y_train, X_valid, y_valid)['accuracy_valid_test']
		#or
		#it = 0
		#accuracies_scores=[]
		#for d in decoded:
		#	accuracies_scores0 = itemgetter('accuracy_valid_test', 'accuracy_train')(objective(d, X_train, y_train, X_valid, y_valid))
		#	accuracies_scores.append(accuracies_scores0)
		#	print(str(it))
		#	it = it+1
        #
        
		accuracies_valid_test0, accuracies_train0 = map(list, zip(*accuracies_scores))      
		scores = accuracies_valid_test0
        # check for new best solution
		for i in range(n_pop):
			if scores[i] > best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %f" % (gen,  decoded[i], scores[i]))
				temp_acc_train = accuracies_train0[accuracies_valid_test0.index(best_eval)]
				temp_decoded_params = decoded[accuracies_valid_test0.index(best_eval)]
				s = 0
                
        # Keep a track of the best values at each generation
		best_accuracies_valid.append(best_eval)
		best_accuracies_train.append(temp_acc_train) # !! index only catches the first value so in some rare cases the train accuracy could be inaccurate. 
		track_generation.append(gen)
		track_hyperparams.append(temp_decoded_params)
        
		# select parents     
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
				s = s + 1
				print(s)
		
		# replace population
		pop = children
 
		print("generation" +str(gen))
		gen = gen + 1
   
		if s>=(3*n_pop):
			break
		if gen>n_iter:
			break
        
	end = time.time()
	print("Runtime:  " +str(end - start))
               
	return [best, best_eval, best_accuracies_valid, best_accuracies_train, track_generation, track_hyperparams]


#itemgetter(1, 3, 2, 5)(my_dict)

#decode(bounds, n_bits, best)