import pickle
import random

RANDOM_GRAPH_FILEPATH_PICKLE = './datasets/random_graph.pickle'
RANDOM_GRAPH_FILEPATH_TXT = './datasets/random_graph.txt'
EDGE_TO_NODE_RATIO = 1768149.0 / 81306.0
NUM_NODES = 1000
NUM_EDGES = int(NUM_NODES * EDGE_TO_NODE_RATIO)

def id_counter():
    id_counter.counter += 1
    return id_counter.counter
id_counter.counter = -1

class Node:
    def __init__(self):
        self.id = id_counter()
        self.edges = set()

def generate_random_graph():
    """ For each node you need at least one edge. Start with one node. In
    each iteration, create a new node and a new edge. The edge is to connect
    the new node with a random node from the previous node set. After all
    nodes are created, create random edges until S is fulfilled. Make sure
    not to create double edges (for this you can use an adjacency matrix).
    Random graph is done in O(S). """
    root = Node()
    nodes = set([root])
    num_edges = 0

    for i in range(1, NUM_NODES):
        node = Node()
        node.edges.add(random.sample(nodes, 1)[0])
        nodes.add(node)
        num_edges += 1

    # Generate edges until 
    for j in range(num_edges, NUM_EDGES):
        tail, head  = random.sample(nodes, 2)
        trys = 0
        while head in tail.edges:
            tail, head  = random.sample(nodes, 2)
            trys += 1
        tail.edges.add(head)
        num_edges += 1
    
    # Convert our graph to CSR representation by first creating an adjacency
    # matrix and then analyzing it to turn it into a CSR
    adjacency_matrix = [[0] * NUM_NODES for i in range(NUM_NODES)]
    stack = [root]
    visited = set()
    while stack:
        curr = stack.pop()
        if curr not in visited:
            visited.add(curr)
            for node in curr.edges:
                stack.append(node)
                adjacency_matrix[curr.id][node.id] = 1.0

    offset = 0
    csr = [[] for i in range(3)]
    for row in range(len(adjacency_matrix)):
        edges = adjacency_matrix[row]
        outdegree = sum(True if num == 1 else False for num in edges)
        csr[1].append(offset)
        for col in range(len(edges)):
            edges[col] /= outdegree
            if edges[col] > 0:
                csr[0].append(edges[col])
                csr[2].append(col)
                offset += 1
    csr[1].append(offset)

    # Write to txt and pickle
    with open(RANDOM_GRAPH_FILEPATH_TXT, "w") as fp:
        fp.write(' '.join(str(i) for i in csr[0]) + '\n')
        fp.write(' '.join(str(i) for i in csr[1]) + '\n')
        fp.write(' '.join(str(i) for i in csr[2]))
    with open(RANDOM_GRAPH_FILEPATH_PICKLE, "wb") as fp:
        pickle.dump(csr, fp)

if __name__ == "__main__":
    generate_random_graph()
