import math
import os
import pickle
import random
from collections import defaultdict

L_CONSTANT = 1
EPSILON_CONSTANT = 0.1
K_CONSTANT = 2

TWITTER_DATASET_FILEPATH = './datasets/twitter'
TWITTER_DATASET_PICKLE_FILEPATH = './datasets/twitter.pickle'
EDGE_FILE_SUFFIX = '.edges'

class Edge:
    def __init__(self, tail, head):
        assert(type(tail) == str)
        assert(type(head) == str)
        self.tail = tail
        self.head = head

    def set_weight(self, weight):
        assert(type(weight) == float)
        self.weight = weight

def calculate_lambda(n, k, l, e):
    return (8.0 + 2 * e) * n * (l * math.log(n) + math.log(comb(n, k)) + math.log(2)) * e ** -2

def comb(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

	""" Now defunct """
def reachable(start, graph):
    """ Performs a DFS search for all reachable nodes and returns a set of
    all visited """
    visited = set()
    stack = [start] # We are using stack for O(1) append and pop
    while stack:
        curr = stack.pop()
        if curr not in visited:
            visited.add(curr)
            for edge in graph[curr]:
                stack.append(edge.head)
    return visited
	
def width(graph, nodes):
    """ Returns the number of edges in a graph """
    return sum(len(graph[node]) for node in nodes)

	""" Now defunct """
def randomly_pruned_graph(graph):
    """ Randomly prunes a graph by removing an edge with 1 - p(e) probability """
    pruned_graph = defaultdict(list)
    for node, edges in graph.items():
        for edge in edges:
            if random.random() < edge.probability:
                pruned_graph[node].append(edge)
    return pruned_graph

def random_reverse_reachable_set(graph):
    """ Returns a set of reverse reachable nodes from a random seed node """
	n = len(graph)
	start = random.choice(graph.keys())
	stack = [start]
	visited = dict()
	while stack:
		curr = stack.pop()
		if curr not in visited:
			visited.add(curr)
			for edge in graph[curr]:
				if random.random() < edge.probability:
					stack.append(edge.head)
					visited[curr].append(edge)
	return visited
					

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
            
def node_selection_experimental(graph, k, theta):
    # Initialize empty set R
    R = {}
    # Generate theta random RR sets and insert them into R
    for i in range(theta):
        R[i] = random_reverse_reachable_set(graph)
    # Initialize a empty node set S_k
    S_k = []
	
	R_used = {}
    for j in range(k):
        # Identify node v_j that covers the most RR sets in R
        v_j, sets_to_remove = find_most_common_node(R)
        # Add v_j into S_k
        S_k.append(v_j)
        # Remove from R all RR sets that are covered by v_j
        for set_id in sets_to_remove:
			R_used = R[set_id]
            del R[set_id]
    return S_k, R, R_used
	
def phase_3_experimental(R,R_used,S_k,k,max_iter):
	go = True
	count = 0
	
	while go:
		# Calculate marginal contributions and which sets make them up
		marginal_contribution = defaultdict(int)
		marginal_count = defaultdict(int)
		seeds_per_RR = defaultdict(int)
		for set_id, rr_set in R_used.items():
			num_seeds = 0
			unique_seed = -1
			for node in rr_set:
				if node in S_k:
					num_seeds += 1
					unique_seed = node
					if num_seeds > 1:
						break
			if num_seeds == 1:
				marginal_contribution[unique_seed].append[set_id]
				if unique_seed in marginal_count:
					marginal_count[unique_seed] += 1
				else:
					marginal_count[unique_seed] = 0
		# Marginal number of RR sets each seed provides is tabulated
		# Now select one seed to return and add its sets in marginal_contribution
		# back into R before finding a new k^th seed
		
		leaving_seed = min(marginal_count, key = marginal_count.get)
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
		
		count += 1
		
		if node == leaving_seed || count >= max_iter:
			go = False
	return S_k
		

	
def node_selection(graph, k, theta):
    # Initialize empty set R
    R = {}
    # Generate theta random RR sets and insert them into R
    for i in range(theta):
        R[i] = random_reverse_reachable_set(graph)
    # Initialize a empty node set S_k
    S_k = []
    for j in range(k):
        # Identify node v_j that covers the most RR sets in R
        v_j, sets_to_remove = find_most_common_node(R)
        # Add v_j into S_k
        S_k.append(v_j)
        # Remove from R all RR sets that are covered by v_j
        for set_id in sets_to_remove:
            del R[set_id]
    return S_k
	
	
	
def kpt_estimation(graph, k):
    n = len(graph)
    m = sum(len(edges) for edges in graph.values())
    for i in range(1, math.log(n, 2)):
        ci = 6 * L_CONSTANT * math.log(n)  + 6 * math.log(math.log(n, 2)) * 2**i
        cum_sum = 0
        for j in range(1, ci + 1):
            r = random_reverse_reachable_set(graph)
            w_r = width(graph, r)
            k_r = 1 - (1 - (w_r / m))**k
            cum_sum += k_r
        if (cum_sum / ci) > 1/(2**i):
            return n * cum_sum / (2 * ci)
    return 1.0

def find_k_seeds(graph, k):
    kpt = kpt_estimation(graph, k)
    lambda_var = calculate_lambda(len(graph), k, L_CONSTANT, EPSILON_CONSTANT)
    theta = lambda_var / kpt
    return node_selection(graph, k ,theta)

def parse_twitter_dataset():
    """ TODO: Our graph is represented as a mapping of node to its list of Edges. We 
    want this graph to represent G_t """
    if os.path.isfile(TWITTER_DATASET_PICKLE_FILEPATH):
        with open(TWITTER_DATASET_PICKLE_FILEPATH, "rb") as fp:
            G_t = pickle.load(fp)
    else:
        G_t = defaultdict(list)
        counter = 0
        for filename in os.listdir(TWITTER_DATASET_FILEPATH):
            if filename.endswith(EDGE_FILE_SUFFIX):
                focal_node = filename.split('.')[0]
                filepath = os.path.join(TWITTER_DATASET_FILEPATH, filename)
                with open(filepath, "r") as fp:
                    print("Processing {}: {}".format(filename, counter))
                    counter += 1
                    line = fp.readline()
                    while line:
                        tail, head = line.split()
                        G_t[head].append(Edge(head, tail))
                        G_t[head].append(Edge(head, focal_node))
                        G_t[tail].append(Edge(tail, focal_node))
                        line = fp.readline()
        """ we first identify the node v that e points to, and then set p(e) = 1/i,
        where i denotes the in-degree of v. So this is effectively the outdegree
        of the tranpose graph"""
        count = 0
        for edges in G_t.values():
            count += 1
            print(count)
            for edge in edges:
                edge.set_weight(1.0/len(edges))
        pickle.dump(G_t, open(TWITTER_DATASET_PICKLE_FILEPATH, 'wb'))
    return G_t

if __name__ == "__main__":
    graph = parse_twitter_dataset()
    print(find_k_seeds(graph, K_CONSTANT))