import pickle
from collections import defaultdict
import sys

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Please provide filepath to dataset in two column node-to-node edge format")
        sys.exit(0)

    graph = defaultdict(set)
    with open(sys.argv[1], "r") as fp:
        lines = fp.read().split('\n')
    nodes = set()
    for line in lines:
        try:
            a, b = line.split()
            graph[a].add(b)
            nodes.add(a)
            nodes.add(b)
        except ValueError:
            pass

    csr = [[], [], []]
    id_vertex_map = {}
    vertex_id_map = {}
    count = 0
    for vertex in nodes:
        vertex_id_map[vertex] = count
        id_vertex_map[count] = vertex
        count += 1

    row_index = 0
    for i in range(count):
        csr[1].append(row_index)
        vertex = id_vertex_map[i]
        for neighbor in graph[vertex]:
            id = vertex_id_map[neighbor]
            csr[0].append(1.0 / len(graph[vertex]))
            csr[2].append(id)
        row_index += len(graph[vertex])
    csr[1].append(len(csr[0]))

    with open(sys.argv[1] + '.pickle', 'wb') as fp:
        pickle.dump(csr, fp)

    with open(sys.argv[1] + '.csv', 'wb') as fp:
        fp.write(" ".join(str(i) for i in csr[0]) + '\n')
        fp.write(" ".join(str(i) for i in csr[1]) + '\n')
        fp.write(" ".join(str(i) for i in csr[2]))
