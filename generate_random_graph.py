import math
import pickle
import random

counter = -1
DATASETS_FILEPATH = './datasets/'

def generate_filepath_txt(num_nodes):
    filename = 'random_graph_' + str(num_nodes)
    return '{}{}.txt'.format(DATASETS_FILEPATH, filename)

def generate_filepath_pickle(num_nodes):
    filename = 'random_graph_' + str(num_nodes)
    return '{}{}.pickle'.format(DATASETS_FILEPATH, filename)

def id_counter():
    global counter
    counter += 1
    return counter

class Node:
    def __init__(self):
        self.id = id_counter()
        self.edges = set()

def generate_random_graph(num_nodes):
    """ For each node you need at least one edge. Start with one node. In
    each iteration, create a new node and a new edge. The edge is to connect
    the new node with a random node from the previous node set. After all
    nodes are created, create random edges until S is fulfilled. Make sure
    not to create double edges (for this you can use an adjacency matrix).
    Random graph is done in O(S). """
    root = Node()
    nodes = set([root])
    edge_count = 0
    num_edges = int(math.log(num_nodes, 1.7)) * num_nodes

    for i in range(1, num_nodes):
        node = Node()
        node.edges.add(random.sample(nodes, 1)[0])
        nodes.add(node)
        edge_count += 1

    # Generate edges until 
    for j in range(edge_count, num_edges):
        tail, head  = random.sample(nodes, 2)
        while head in tail.edges:
            tail, head  = random.sample(nodes, 2)
        tail.edges.add(head)
        edge_count += 1
    
    # Convert our graph to CSR representation by first creating an adjacency
    # matrix and then transforming it to a CSR

    # Generating adjacency matrix
    adjacency_matrix = [[0] * num_nodes for i in range(num_nodes)]
    stack = [root]
    visited = set()
    while stack:
        curr = stack.pop()
        if curr not in visited:
            visited.add(curr)
            for node in curr.edges:
                stack.append(node)
                adjacency_matrix[curr.id][node.id] = 1.0

    # Adjacency matrix -> CSR
    offset = 0
    csr = [[] for i in range(3)]
    for row in range(len(adjacency_matrix)):
        edges = adjacency_matrix[row]
        outdegree = sum(edges)
        csr[1].append(offset)
        for col in range(len(edges)):
            if outdegree > 0:
                edges[col] /= outdegree
                if edges[col] > 0:
                    csr[0].append(edges[col])
                    csr[2].append(col)
                    offset += 1
    csr[1].append(offset)

    # Write to txt and pickle
    with open(generate_filepath_txt(num_nodes), "w") as fp:
        fp.write(' '.join(str(i) for i in csr[0]) + '\n')
        fp.write(' '.join(str(i) for i in csr[1]) + '\n')
        fp.write(' '.join(str(i) for i in csr[2]))
    with open(generate_filepath_pickle(num_nodes), "wb") as fp:
        pickle.dump(csr, fp)

if __name__ == "__main__":
    for num_nodes in range(20,101):
        generate_random_graph(num_nodes)
        counter = -1
