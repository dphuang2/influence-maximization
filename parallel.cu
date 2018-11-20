#include "shared.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>

using namespace std;

typedef struct node
{
    int id;
    node *prev;
    node *next;
    __device__ node(int id) : id(id){};
} node_t;

__global__ void init_rng(int nthreads, curandState *states, unsigned long long seed, unsigned long long offset)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nthreads)
        return;
    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(seed + id, 0, offset, &states[id]);
}

__global__ void generate_rr_sets(float *data, int *rows, int *cols, bool *out, int *nodeHistogram, int numNodes, int numSets, curandState *states)
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numSets)
    {
        curandState *state = &states[tid];

        /* Because C does not give us the luxury of dynamic arrays, to imitate the
        behavior of a stack, I am using a linked list*/
        int randomNodeId = ceil(numNodes * curand_uniform(state)) - 1;
        node *stack = new node(randomNodeId);
        node *auxiliary = new node(AUXILIARY_NODE_ID);
        auxiliary->next = stack;
        stack->prev = auxiliary;

        // Returns false when stack is NULL
        while (stack != NULL && stack->id != AUXILIARY_NODE_ID)
        {
            // pop from stack
            int currentNodeId = stack->id;
            node *temp = stack;
            stack = stack->prev;
            free(temp);

            // If current is not in visited
            if (!out[tid * numNodes + currentNodeId])
            {
                out[tid * numNodes + currentNodeId] = true;
                atomicAdd(&nodeHistogram[currentNodeId], 1);

                int dataStart = rows[currentNodeId];
                int dataEnd = rows[currentNodeId + 1];

                for (unsigned int i = dataStart; i < dataEnd; i++)
                {
                    if (curand_uniform(state) < data[i])
                    {
                        // append to stack
                        stack->next = new node(cols[i]);
                        stack->next->prev = stack;
                        stack = stack->next;
                    }
                }
            }
        }
        free(auxiliary);
    }
}

__global__ void count_node_to_node_intersections(int *counts, bool *batch, int num_rows, int num_nodes)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid % num_nodes;
    int node_y = (tid / num_nodes) % num_nodes;
    int node_z = tid / (num_nodes * num_nodes);

    if (row < num_rows && node_y < num_nodes && node_z < num_nodes)
    {
        if (batch[row * num_nodes + node_y] && batch[row * num_nodes + node_z])
        {
            atomicAdd(&counts[node_y * num_nodes + node_z], 1);
        }
    }
}

__global__ void update_counts(int *intersections, int *histogram, int numNodes, int nodeToDelete)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= numNodes || col >= numNodes)
        return;
    if (row >= col)
    {
        intersections[row * numNodes + col] -= intersections[nodeToDelete * numNodes + col];
        if (row == col)
        {
            histogram[row] = intersections[row * numNodes + col];
        }
    }
    else if (row < col)
    {
        intersections[row * numNodes + col] -= intersections[row * numNodes + nodeToDelete];
    }
}

unordered_set<int> nodeSelection(CSR *graph, int k, double theta)
{
    unordered_set<int>::iterator it;
    unordered_set<int> seeds;
    map<int, unordered_set<int>> R;
    size_t freeGPUBytes;
    size_t totalGPUBytes;
    float *deviceData;
    int *deviceRows;
    int *deviceCols;
    int *deviceNodeHistogram;
    int *deviceNodeToNodeIntersections;
    bool *deviceProcessedRows;
    curandState *deviceStates;

    // Initialize data, rows, and cols
    int sizeOfData = graph->data.size() * sizeof(float);
    int sizeOfRows = graph->rows.size() * sizeof(int);
    int sizeOfCols = graph->cols.size() * sizeof(int);
    cudaMalloc((void **)&deviceData, sizeOfData);
    cudaMalloc((void **)&deviceRows, sizeOfRows);
    cudaMalloc((void **)&deviceCols, sizeOfCols);
    cudaMemcpy(deviceData, &(graph->data[0]), sizeOfData, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceRows, &(graph->rows[0]), sizeOfRows, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceCols, &(graph->cols[0]), sizeOfCols, cudaMemcpyHostToDevice);

    // Initialize output of kernel
    int numNodes = (int)graph->rows.size() - 1;
    int sizeOfNodeHistogram = sizeof(int) * numNodes;
    int sizeOfNodeToNodeIntersections = sizeof(int) * numNodes * numNodes;
    cudaMalloc((void **)&deviceNodeHistogram, sizeOfNodeHistogram);
    cudaMalloc((void **)&deviceNodeToNodeIntersections, sizeOfNodeToNodeIntersections);
    cudaMemset(deviceNodeHistogram, 0, sizeOfNodeHistogram);
    cudaMemset(deviceNodeToNodeIntersections, 0, sizeOfNodeToNodeIntersections);

    // Calculate number of batches
    cudaMemGetInfo(&freeGPUBytes, &totalGPUBytes);
    int numRowsPerBatch = ceil(
        ((freeGPUBytes / 1.5) - (4 * numNodes + pow(numNodes, 2))) /
        (numNodes + sizeof(curandState)));
    int numBatches = ceil(theta / numRowsPerBatch);

    // Initialize processed rows output
    long long int sizeOfProcessedRows = sizeof(bool) * numRowsPerBatch * numNodes;
    cudaMalloc((void **)&deviceProcessedRows, sizeOfProcessedRows);

    // Initialize RNG States
    cudaMalloc((void **)&deviceStates, numRowsPerBatch * sizeof(curandState));
    dim3 dimGrid(ceil(float(numRowsPerBatch) / BLOCK_SIZE), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    init_rng<<<dimGrid, dimBlock>>>(numRowsPerBatch, deviceStates, 1, 0);
    cudaDeviceSynchronize();

    // Process batches
    int numRowsProcessed = 0;
    for (int i = 0; i < numBatches; i++)
    {
        cudaMemset(deviceProcessedRows, false, sizeOfProcessedRows);
        int numRowsToProcess = min(numRowsPerBatch, (int)ceil(theta) - numRowsProcessed);
        dimGrid = dim3(ceil(float(numRowsToProcess) / BLOCK_SIZE), 1, 1);
        dimBlock = dim3(BLOCK_SIZE, 1, 1);
        generate_rr_sets<<<dimGrid, dimBlock>>>(deviceData, deviceRows, deviceCols, deviceProcessedRows, deviceNodeHistogram, numNodes, numRowsToProcess, deviceStates);
        cudaDeviceSynchronize();

        dimGrid = dim3(ceil(double(numRowsToProcess * numNodes) / BLOCK_SIZE), 1, 1);
        dimBlock = dim3(BLOCK_SIZE, 1, 1);
        count_node_to_node_intersections<<<dimGrid, dimBlock>>>(deviceNodeToNodeIntersections, deviceProcessedRows, numRowsToProcess, numNodes);
        cudaDeviceSynchronize();
    }

    thrust::device_ptr<int> dev_ptr(deviceNodeHistogram);
    for (int j = 0; j < k; j++)
    {
        int mostCommonNode = (thrust::max_element(dev_ptr, dev_ptr + numNodes) - dev_ptr);
        seeds.insert(mostCommonNode);
        dimGrid = dim3(ceil(float(numNodes) / TILE_X_2D), ceil(float(numNodes) / TILE_Y_2D), 1);
        dimBlock = dim3(TILE_X_2D, TILE_Y_2D, 1);
        update_counts<<<dimGrid, dimBlock>>>(deviceNodeToNodeIntersections, deviceNodeHistogram, numNodes, mostCommonNode);
    }

    return seeds;
}

unordered_set<int> findKSeeds(CSR *graph, int k)
{
    double n = double(graph->rows.size() - 1);
    struct timeval t1;
    struct timeval t2;

    gettimeofday(&t1, NULL);
    double kpt = kptEstimation(graph, k);
    gettimeofday(&t2, NULL);
    printf("kptEstimation: %ld\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec - t1.tv_usec));

    gettimeofday(&t1, NULL);
    double lambda = calculateLambda(n, k, L_CONSTANT, EPSILON_CONSTANT);
    gettimeofday(&t2, NULL);
    printf("calculateLambda: %ld\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec - t1.tv_usec));

    double theta = lambda / kpt;

    gettimeofday(&t1, NULL);
    unordered_set<int> selectedNodes = nodeSelection(graph, k, theta);
    gettimeofday(&t2, NULL);
    printf("selectedNodes: %ld\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec - t1.tv_usec));

    return selectedNodes;
}

int main(int argc, char **argv)
{
    if (!fileExists(RANDOM_GRAPH_FILEPATH))
    {
        cout << "File " << RANDOM_GRAPH_FILEPATH << " did not exist...exiting" << endl;
        exit(1);
    }

    // Creating an object of CSVWriter
    CSVReader reader(RANDOM_GRAPH_FILEPATH);
    // Get the data from CSV File
    CSR *graph = covertToCSR(reader.getData());

    struct timeval t1, t2;
    for (int i = 0; i < 1; i++)
    {
        gettimeofday(&t1, NULL);
        unordered_set<int> seeds = findKSeeds(graph, K_CONSTANT);
        gettimeofday(&t2, NULL);
        unordered_set<int>::iterator it;
        for (it = seeds.begin(); it != seeds.end(); it++)
        {
            cout << *it << " ";
        }
        printf("- %ld\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec - t1.tv_usec));
    }
    return 0;
}
