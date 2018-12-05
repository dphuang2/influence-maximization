import math
import operator as op
import os
import sys
import pickle
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from gurobipy import *
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from scipy.special import comb

from generate_random_graph import generate_filepath_pickle
from timer import (cumulative_runtimes, execution_counts,
                   find_k_seeds_runtimes, runtimes, timeit, to_csv)

L_CONSTANT = 1
EPSILON_CONSTANT = 0.2
K_CONSTANT = 5

TWITTER_DATASET_FILEPATH = './datasets/twitter'
TWITTER_DATASET_PICKLE_FILEPATH = './python/twitter.pickle'
GPLUS_DATASET_PICKLE_FILEPATH= './python/gplus.pickle'
EDGE_FILE_SUFFIX = '.edges'
RANDOM_CSR_GRAPH_FILEPATH = './datasets/random_graph.pickle'


@timeit
def node_selection_experimental(graph, k, theta):
    # Initialize empty set R
    R = {}
    # Generate theta random RR sets and insert them into R
    for i in range(math.ceil(theta)):
        R[i] = random_reverse_reachable_set(graph)
    # Initialize a empty node set S_k
    S_k = []

    R_used = {}
    for j in range(k):
	
        v_j, sets_to_remove = find_most_common_node(R)
        # Add v_j into S_k
        S_k.append(v_j)
        # Remove from R all RR sets that are covered by v_j
        for set_id in sets_to_remove:
            R_used[set_id] = R[set_id]
            del R[set_id]
    return S_k, R, R_used

	
@timeit
def construct_BIP(R,k,theta,n):
	# Construct coefficient matrix
	theta = math.ceil(theta)
	coef = np.zeros([theta,n])
	set_present = []
	for i in range(theta):
		set_present.append(i)
		rr_set = R[i]
		for node in rr_set:
			coef[i,node] = 1
	m = Model("BIP Seed Selection")
	
	nodes = []
	for i in range(n):
		nodes.append(i)
	
	
	selected = m.addVars(nodes, vtype=GRB.BINARY, name = "selected")
	present = m.addVars(set_present, vtype=GRB.BINARY, name = "set present")
	
	# set only present if represented
	m.addConstrs( (quicksum(selected[j] * coef[i,j] for j in nodes) >= present[i] for i in set_present), "SetRepresented" )
	
	# at most k seeds
	m.addConstr( quicksum(selected[i] for i in nodes) <= k, "KSeeds" )
	
	# Objective function
	m.setObjective((quicksum(present[i] for i in set_present)), GRB.MAXIMIZE)
	
	m.optimize()
	
	S_k = []
	
	if m.status == GRB.Status.OPTIMAL:
		print(m.status)
		solution = m.getAttr('x',selected)
		for i in nodes:
			if solution[i] > 0:
				S_k.append(i)
	
	return m, S_k
	
	
@timeit
def run_3_phase(graph, k, max_iter):
    S_k, R, R_used, theta = find_k_seeds_IMM(graph,k)
    initial_covered = len(R_used)
    print(S_k,initial_covered)
    S_k, covered, overlapped = phase_3_experimental(R, R_used, S_k, k, max_iter)
    percent_coverage_increase = (1.0*covered)/initial_covered
    return S_k, percent_coverage_increase, overlapped
	
@timeit	
def phase_3_experimental(R, R_used, S_k, k, max_iter):
    go = True
    iterations_performed = 0
    overlapped = 0
    while go:
        # Calculate marginal contributions and which sets make them up
        marginal_contribution = {} # List of sets that only each seed is in
        marginal_count = {} # Number sets only inhabited by each seed
        overlapped = 0
        for seed in S_k:
            marginal_contribution[seed] = []
            marginal_count[seed] = 0
        for set_id in R_used.keys():
            rr_set = R_used[set_id]
            num_seeds = 0
            unique_seed = -1
            for node in rr_set:
                if node in S_k:
                    num_seeds += 1
                    unique_seed = node
                    if num_seeds > 1:
                        overlapped += 1
                        break
            if num_seeds == 1:
                marginal_contribution[unique_seed].append(set_id)
                marginal_count[unique_seed] += 1
        # Marginal number of RR sets each seed provides is tabulated
        # Now select one seed to return and add its sets in marginal_contribution
        # back into R before finding a new k^th seed
        # print(marginal_contribution)
        #leaving_seed = min(marginal_count, key=marginal_count.get)
        leaving_seed = S_k[random.randint(0,k-2)]
        S_k.remove(leaving_seed)
        for set_id in marginal_contribution[leaving_seed]:
            R[set_id] = R_used[set_id]
            del R_used[set_id]

        # Select new k^th seed

        node, sets_to_remove = find_most_common_node(R)
        S_k.append(node)
        for set_id in sets_to_remove:
            R_used[set_id] = R[set_id]
            del R[set_id]

        iterations_performed += 1

        #if node == leaving_seed or iterations_performed >= max_iter:
        if iterations_performed >= max_iter:
            go = False
    covered = len(R_used)
    return S_k, covered, overlapped


@timeit
def width(graph, nodes):
    """ Returns the number of edges in a graph """
    count = 0
    for node in nodes:
        value_start = graph[1][node]
        value_end = graph[1][node + 1]
        count += value_end - value_start
    return count


@timeit
def find_most_common_node(rr_sets):
    counts = defaultdict(int)
    exists_in_set = defaultdict(list)
    maximum = 0
    most_common_node = 0
    for set_id, rr_set in rr_sets.items():
        for node in rr_set:
            exists_in_set[node].append(set_id)
            counts[node] += 1
            if counts[node] > maximum:
                most_common_node = node
                maximum = counts[node]
    return most_common_node, exists_in_set[most_common_node]


@timeit
def node_selection(graph, k, theta, R):
    if not R:
	    # Initialize empty set R	
        R = {}
        # Generate theta random RR sets and insert them into R
        for i in range(int(math.ceil(theta))):
            R[i] = random_reverse_reachable_set(graph)
    # Initialize a empty node set S_k
    S_k = []
    R_temp = R.copy()
    for j in range(k):
        # Identify node v_j that covers the most RR sets in R
        v_j, sets_to_remove = find_most_common_node(R)
        # Add v_j into S_k
        S_k.append(v_j)
        # Remove from R all RR sets that are covered by v_j
        for set_id in sets_to_remove:
            del R[set_id]
    return S_k, (math.ceil(theta) - len(R.keys())), R_temp # Return number of RR sets covered (performance comparison and IMM necessary)

@timeit
def node_selection_BIP(graph,n,k,theta,R):
	if not R:
		R = {}
		for i in range(int(math.ceil(theta))):
			R[i] = random_reverse_reachable_set(graph)
	m, S_k = construct_BIP(R,k,theta,n)
	return m, S_k, R
	

@timeit
def calculate_lambda(n, k, l, e):
    return (8.0 + 2 * e) * n * (l * math.log(n) + math.log(comb(n, k)) + math.log(2)) * e ** -2


# Section from here is IMM algorithm for theta computation

@timeit
def calculate_lambda_prime(n, k, l, eps):
    return (2.0 + 2.0/3.0 * eps) * (math.log(comb(n, k)) + l * math.log(n) * math.log(math.log(n)/math.log(2))) * n * eps ** (-2)

def calculate_lambda_star(n,k,l,eps):
	alpha = math.sqrt(l*math.log(n) + math.log(2))
	beta = math.sqrt( (1-1/math.e) * (math.log(math.factorial(n)/(math.factorial(k)*math.factorial(n-k))) + l*math.log(n)+math.log(2)))
	return 2*n * ((1-1/math.e)*alpha+beta)**2 * eps **(-2)
@timeit	
def find_theta_IMM(graph,n,k,e,l):
	"""Finds the tight lower bound on OPT derived in the IMM algorithm"""
	lb = 1
	eps = e * math.sqrt(2)
	lam = calculate_lambda_prime(n,k,l,eps)
	lamStar = calculate_lambda_star(n,k,l,e)
	iterations = int(math.ceil(math.log(n)/math.log(2) - 1))
	for i in range(iterations):
		x = n/(2.0 ** (i+1))
		theta = lam/x
		S_k, uncovered, place_holder = node_selection(graph,k,theta,{})
		frac = (theta-uncovered) * 1.0/theta
		if n*frac >= (1+eps)*x:
			return lamStar / (n * frac / (1+eps))
	return lamStar
	
# Section for IMM algorithm finished
	
@timeit
def random_reverse_reachable_set(graph):
    """ Returns a set of reverse reachable nodes from a random seed node """
    n = len(graph[1]) - 1
    start = random.choice(range(n))
    stack = [start]
    visited = set()
    while stack:
        curr = stack.pop()
        if curr not in visited:
            visited.add(curr)

            # We are getting the value offsets from the second array here
            value_start = graph[1][curr]
            value_end = graph[1][curr + 1]

            # Using the offsets we just extracted, get head of outgoing edges
            edges = graph[2][value_start:value_end]

            # Do the same with the values of the edges
            probabilities = graph[0][value_start:value_end]

            # assert for correctness
            assert(len(edges) == len(probabilities))

            for i in range(len(edges)):
                if random.random() < probabilities[i]:
                    stack.append(edges[i])
    return visited


@timeit
def kpt_estimation(graph, k):
    n = len(graph[1]) - 1
    m = len(graph[0])
    for i in range(1, int(math.log(n, 2))):
        ci = 6 * L_CONSTANT * math.log(n) + 6 * math.log(math.log(n, 2)) * 2**i
        cum_sum = 0
        for j in range(int(ci)):
            R = random_reverse_reachable_set(graph)
            w_r = width(graph, R)
            k_r = 1 - (1 - (w_r / m))**k
            cum_sum += k_r
        if (cum_sum / ci) > 1/(2**i):
            return n * cum_sum / (2 * ci)
    return 1.0


@timeit
def find_k_seeds(graph, k):
    kpt = kpt_estimation(graph, k)
    lambda_var = calculate_lambda(
        len(graph[1]) - 1, k, L_CONSTANT, EPSILON_CONSTANT)
    theta = lambda_var / kpt
    return node_selection(graph, k, theta)

@timeit
def find_k_seeds_IMM(graph, k):
	n = len(graph[1]) - 1
	theta = find_theta_IMM(graph,n,k,EPSILON_CONSTANT,L_CONSTANT)
	#S_k, R, R_used = node_selection_experimental(graph, k, theta)
	S_k2, covered2, R = node_selection(graph,k,theta,{})
	print(theta)
	print(covered2)
	print(["IMM coverage",(covered2*1.0)/(math.ceil(theta))])
	m, S_k1, R = node_selection_BIP(graph,n,k,theta,R)
	covered1 = m.objval
	print(["BIP coverage percentage increase", (covered1-covered2)/(1.0*covered2)*100])
	return m, S_k1, covered1, S_k2, covered2

if __name__ == "__main__":
    for i in range(5000, 5001):
        for j in range(1):
            #graph = pickle.load(open('./python/random_graph_5000.pickle', "rb"))
            print(generate_filepath_pickle(i))
            print(TWITTER_DATASET_PICKLE_FILEPATH)
            graph = pickle.load(open(GPLUS_DATASET_PICKLE_FILEPATH, "rb"))
            print(find_k_seeds_IMM(graph, K_CONSTANT))

    with open("execution_counts.csv", "w") as fp:

        fp.write(to_csv(execution_counts))

    with open("cumulative_runtimes.csv", "w") as fp:

        fp.write(to_csv(cumulative_runtimes))

    with open("find_k_seeds_runtimes.csv", "w") as fp:

        fp.write(to_csv(find_k_seeds_runtimes))

    with open("runtimes.csv", "w") as fp:

        fp.write(to_csv(runtimes))
