import math
import operator as op
import os
import pickle
import random
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from scipy.special import comb

from generate_random_graph import generate_filepath_pickle
from pycuda import (autoinit, characterize, compiler, curandom, driver,
                    gpuarray, tools)
from timer import (cumulative_runtimes, execution_counts,
                   find_k_seeds_runtimes, runtimes, timeit, to_csv)

L_CONSTANT = 1
EPSILON_CONSTANT = 0.2
K_CONSTANT = 2
BLOCK_SIZE = 1024
TILE_X = 1
TILE_Y = 32
TILE_Z = 32

SIZEOF_GENERATOR = characterize.sizeof(
    'curandStateXORWOW', '#include <curand_kernel.h>')

TWITTER_DATASET_FILEPATH = './datasets/twitter'
TWITTER_DATASET_PICKLE_FILEPATH = './datasets/twitter.pickle'
EDGE_FILE_SUFFIX = '.edges'
RANDOM_CSR_GRAPH_FILEPATH = './datasets/random_graph.pickle'
GENERATE_RR_SETS_CUDA_CODE_FILEPATH = 'node_selection.cu'

# Compile kernel code
with open(GENERATE_RR_SETS_CUDA_CODE_FILEPATH, "r") as fp:
    content = fp.read()
mod = compiler.SourceModule(content, no_extern_c=True)


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
def node_selection(graph, k, theta):
    theta = math.ceil(theta)

    generate_rr_sets = mod.get_function('generate_rr_sets')
    count_node_to_node_intersections = mod.get_function(
        'count_node_to_node_intersections')

    # Initialize all the necessary device memory
    data_cpu = np.asarray(graph[0]).astype(np.float32)
    rows_cpu = np.asarray(graph[1]).astype(np.int32)
    cols_cpu = np.asarray(graph[2]).astype(np.int32)
    data_gpu = driver.mem_alloc(data_cpu.size * data_cpu.dtype.itemsize)
    rows_gpu = driver.mem_alloc(rows_cpu.size * rows_cpu.dtype.itemsize)
    cols_gpu = driver.mem_alloc(cols_cpu.size * cols_cpu.dtype.itemsize)
    driver.memcpy_htod(data_gpu, data_cpu)
    driver.memcpy_htod(rows_gpu, rows_cpu)
    driver.memcpy_htod(cols_gpu, cols_cpu)

    num_nonzeros = np.int32(len(graph[0]))
    num_nodes = np.int32(len(graph[1]) - 1)

    # Calculate the number of batches by using half of our 3/4 of our RAM per batch
    bytes_left, _ = driver.mem_get_info()
    num_rows_per_batch = math.ceil(
        ((bytes_left / 1.5) - (4 * (num_nodes + num_nodes**2))) / (num_nodes + SIZEOF_GENERATOR))
    num_batches = math.ceil(theta / num_rows_per_batch)
    node_histogram = gpuarray.empty(num_nodes, dtype=np.int32)
    # node_to_node_intersections = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    node_to_node_intersections = gpuarray.empty(
        (num_nodes, num_nodes), np.int32)

    # Process the batches
    num_rows_processed = 0
    for _ in range(num_batches):

        # Calculate number of rows to process
        num_rows_to_process = np.int32(min(
            num_rows_per_batch, theta - num_rows_processed))

        # Only initialize a new rng_state array if its our first run
        if 'rng_states' not in locals():
            rng_states = get_rng_states(num_rows_to_process)

        # Initialize output of kernel if we don't have it yet, otherwise just reset it with False values
        if 'processed_rows' not in locals():
            processed_rows = gpuarray.empty((num_rows_to_process, num_nodes), dtype=np.bool_)
        else:
            processed_rows.fill(False)

        # Define number of blocks
        dim_grid = (math.ceil(num_rows_to_process / BLOCK_SIZE), 1, 1)
        dim_block = (BLOCK_SIZE, 1, 1)

        # Launch kernel to generate RR sets
        generate_rr_sets(data_gpu, rows_gpu, cols_gpu, processed_rows, node_histogram, num_nodes,
                         num_nonzeros, num_rows_to_process, rng_states, grid=dim_grid, block=dim_block)

        # Counts node to node intersections
        dim_grid = (math.ceil(float(num_rows_to_process) / TILE_X),
                    math.ceil(float(num_nodes) / TILE_Y), math.ceil(float(num_nodes) / TILE_Z))
        dim_block = (TILE_X, TILE_Y, TILE_Z)
        count_node_to_node_intersections(
            node_to_node_intersections, processed_rows, num_rows_to_process, num_nodes, grid=dim_grid, block=dim_block)

    # Initialize a empty node set S_k
    S_k = []
    for _ in range(k):
        # Identify node that covers the most RR sets in R
        node = np.argmax(node_histogram.get())
        # Add node into S_k
        S_k.append(node)
        # Remove from R all RR sets that are covered by v_j
        for intersecting_node in range(num_nodes):
            node_histogram[intersecting_node] -= node_to_node_intersections[node][intersecting_node]
    return S_k


def get_rng_states(size, seed=1):
    "Return `size` number of CUDA random number generator states."
    rng_states = driver.mem_alloc(np.long(size*SIZEOF_GENERATOR))

    init_rng = mod.get_function('init_rng')

    init_rng(np.int32(size), rng_states, np.uint64(seed),
             np.uint64(0), block=(BLOCK_SIZE, 1, 1), grid=(math.ceil(float(size)/BLOCK_SIZE), 1))

    return rng_states


@timeit
def calculate_lambda(n, k, l, e):
    return (8.0 + 2 * e) * n * (l * math.log(n) + math.log(comb(n, k)) + math.log(2)) * e ** -2


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
        for _ in range(int(ci)):
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


if __name__ == "__main__":
    for i in range(100, 101):
        for j in range(1):
            graph = pickle.load(open(generate_filepath_pickle(i), "rb"))
            print(find_k_seeds(graph, K_CONSTANT))

    with open("execution_counts.csv", "w") as fp:

        fp.write(to_csv(execution_counts))

    with open("cumulative_runtimes.csv", "w") as fp:

        fp.write(to_csv(cumulative_runtimes))

    with open("find_k_seeds_runtimes.csv", "w") as fp:

        fp.write(to_csv(find_k_seeds_runtimes))

    with open("runtimes.csv", "w") as fp:

        fp.write(to_csv(runtimes))
