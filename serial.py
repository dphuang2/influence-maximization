import math
import random
from collections import defaultdict

L_CONSTANT = 1
EPSILON_CONSTANT = 0.1
K_CONSTANT = 2

class Edge:
    def __init__(self, tail, head, probability):
        assert(type(tail) == int)
        assert(type(head) == int)
        assert(type(probability) == float)
        self.tail = tail
        self.head = head
        self.probability = probability

def calculate_lambda(n, k, l, e):
    # λ = (8 + 2ε)n · (ℓ log n + log nCk  + log 2) · ε ^ -2
    return (8.0 + 2 * e) * n * (l * math.log(n) + math.log(comb(n, k)) + math.log(2)) * e ** -2

def comb(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def reachable(start, graph):
    """ Performs a DFS search for all reachable nodes and returns a set of all visited """
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
    pruned_graph = randomly_pruned_graph(graph)
    return reachable(pruned_graph, random.choice(pruned_graph.keys()))

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
    lambda_var = calculate_lambda(len(graph.keys()), k, L_CONSTANT, EPSILON_CONSTANT)
    theta = lambda_var / kpt
    return node_selection(graph, k ,theta)

def parse_dataset(filename):
    """ TODO: Our graph is represented as a mapping of node to its list of Edges. We 
    want this graph to represent G_t """
    return {}

if __name__ == "__main__":
    graph = parse_dataset("")
    print(find_k_seeds(graph, K_CONSTANT))