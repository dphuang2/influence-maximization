#include "shared.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>

using namespace std;

typedef struct Node {
    int id;
    Node *prev;
    Node *next;
    __device__ Node(int id) : id(id) {};
} Node_t;

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
    if (tid < numSets) {
        curandState *state = &states[tid];

        /* Because C does not give us the luxury of dynamic arrays, to imitate the
        behavior of a stack, I am using a linked list*/
        int randomNodeId = ceil(numNodes * curand_uniform(state)) - 1;
        Node *stack = new Node(randomNodeId);
        Node *auxiliary = new Node(AUXILIARY_NODE_ID);
        auxiliary->next = stack;
        stack->prev = auxiliary;

        // Returns false when stack is NULL
        while (stack != NULL && stack->id != AUXILIARY_NODE_ID) {
            // pop from stack
            int currentNodeId = stack->id;
            Node *temp = stack;
            stack = stack->prev;
            free(temp);

            // If current is not in visited
            if (!out[tid * numNodes + currentNodeId]) {
                out[tid * numNodes + currentNodeId] = true;
                atomicAdd(&nodeHistogram[currentNodeId], 1);

                int dataStart = rows[currentNodeId];
                int dataEnd = rows[currentNodeId + 1];

                for (unsigned int i = dataStart; i < dataEnd; i++) {
                    if (curand_uniform(state) < data[i]) {
                        // append to stack
                        stack->next = new Node(cols[i]);
                        stack->next->prev = stack;
                        stack = stack->next;
                    }
                }
            }
        }
        free(auxiliary);
    }
}

__global__ void update_counts(bool * data, int * rows, int * cols, int * histogram, int numRows, int numNodes, int nodeToDelete)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= numRows)
        return;

    int dataStart = rows[row];
    int dataEnd = rows[row + 1];

    bool exists = false;
    // Lets first figure out if this set includes the delete node and still exists
    for (int i = dataStart; i < dataEnd; i++) {
        if (data[i] && cols[i] == nodeToDelete) {
            exists = true;
            break;
        }
    }

    if (exists) {
        // Now lets set all of them to false and update histogram concurrently
        for (int i = dataStart; i < dataEnd; i++) {
            data[i] = false;
            atomicSub(&histogram[cols[i]], 1);
        }
    }
}

unordered_set<int> nodeSelection(CSR<float> *graph, int k, double theta)
{
    unordered_set<int>::iterator it;
    unordered_set<int> seeds;
    map<int, unordered_set<int>> R;
    size_t freeGPUBytes;
    size_t totalGPUBytes;
    float * deviceDataFloat;
    bool * deviceDataBool;
    int * deviceRows;
    int * deviceCols;
    int * deviceNodeHistogram;
    bool * deviceProcessedRows;
    bool * hostProcessedRows;
    curandState * deviceStates;

    // Initialize data, rows, and cols
    int sizeOfData = graph->data.size() * sizeof(float);
    int sizeOfRows = graph->rows.size() * sizeof(int);
    int sizeOfCols = graph->cols.size() * sizeof(int);
    cudaMalloc((void **) &deviceDataFloat, sizeOfData);
    cudaMalloc((void **) &deviceRows, sizeOfRows);
    cudaMalloc((void **) &deviceCols, sizeOfCols);
    cudaMemcpy(deviceDataFloat, &(graph->data[0]), sizeOfData, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceRows, &(graph->rows[0]), sizeOfRows, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceCols, &(graph->cols[0]), sizeOfCols, cudaMemcpyHostToDevice);

    // Initialize output of kernel
    int numNodes = (int) graph->rows.size() - 1;
    int sizeOfNodeHistogram = sizeof(int) * numNodes;
    cudaMalloc((void **) &deviceNodeHistogram, sizeOfNodeHistogram);
    cudaMemset(deviceNodeHistogram, 0, sizeOfNodeHistogram);

    // Calculate number of batches
    cudaMemGetInfo(&freeGPUBytes, &totalGPUBytes);
    int numRowsPerBatch = ceil(
            ((freeGPUBytes / 1.5) - (4 * numNodes + pow(numNodes, 2)))
            /
            (numNodes + sizeof(curandState)));
    int numBatches = ceil(theta / numRowsPerBatch);

    // Initialize processed rows output
    long long int sizeOfProcessedRows = sizeof(bool) * numRowsPerBatch * numNodes;
    cudaMalloc((void **) &deviceProcessedRows, sizeOfProcessedRows);
    hostProcessedRows = (bool *) malloc(sizeOfProcessedRows);

    // Initialize RNG States
    cudaMalloc((void **) &deviceStates, numRowsPerBatch * sizeof(curandState));
    dim3 dimGrid(ceil(float(numRowsPerBatch) / BLOCK_SIZE), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    init_rng<<<dimGrid, dimBlock>>>(numRowsPerBatch, deviceStates, 1, 0);
    cudaDeviceSynchronize();

    // Process batches
    int numRowsProcessed = 0;
    // Can't use bool: https://stackoverflow.com/questions/8399417/why-vectorboolreference-doesnt-return-reference-to-bool
    CSR<char> * processedRows = new CSR<char>();
    for (int i = 0; i < numBatches; i++) {
        // Reset processed rows for new kernel
        cudaMemset(deviceProcessedRows, false, sizeOfProcessedRows);
        
        // Process the minimum number of rows
        int  numRowsToProcess = min(numRowsPerBatch, (int) ceil(theta) - numRowsProcessed);

        // Launch RR generation kernel
        dimGrid = dim3(ceil(float(numRowsToProcess) / BLOCK_SIZE), 1, 1);
        dimBlock = dim3(BLOCK_SIZE, 1, 1);
        generate_rr_sets<<<dimGrid, dimBlock>>>(deviceDataFloat, deviceRows, deviceCols, deviceProcessedRows, deviceNodeHistogram, numNodes, numRowsToProcess, deviceStates);
        cudaDeviceSynchronize();

        // Add to our running processedRows CSR using the rows and cols members
        cudaMemcpy(hostProcessedRows, deviceProcessedRows, sizeOfProcessedRows, cudaMemcpyDeviceToHost);
        for (int j = 0; j < numRowsToProcess; j++) {
            processedRows->rows.push_back(processedRows->data.size());
            for (int k = 0; k < numNodes; k++) {
                if (hostProcessedRows[(unsigned long long int) j * numNodes + k]) {
                    processedRows->data.push_back(true);
                    processedRows->cols.push_back(k);
                }
            }
        }
        processedRows->rows.push_back(processedRows->data.size());

        numRowsProcessed += numRowsToProcess;
    }

    // Initialize device pointer for max element reduction
    thrust::device_ptr<int> dev_ptr(deviceNodeHistogram);

    // Copy our processedRows CSR to device
    cudaFree(deviceRows);
    cudaFree(deviceCols);
    sizeOfData = processedRows->data.size() * sizeof(char);
    sizeOfRows = processedRows->rows.size() * sizeof(int);
    sizeOfCols = processedRows->cols.size() * sizeof(int);
    cudaMalloc((void **) &deviceDataBool, sizeOfData);
    cudaMalloc((void **) &deviceRows, sizeOfRows);
    cudaMalloc((void **) &deviceCols, sizeOfCols);
    cudaMemcpy(deviceDataBool, &(processedRows->data[0]), sizeOfData, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceRows, &(processedRows->rows[0]), sizeOfRows, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceCols, &(processedRows->cols[0]), sizeOfCols, cudaMemcpyHostToDevice);

    // Initialize dimensions for count updating
    int mostCommonNode;
    dimGrid = dim3(ceil(float(numRowsProcessed) / BLOCK_SIZE), 1, 1);
    dimBlock = dim3(BLOCK_SIZE, 1, 1);
    // Select nodes using histogram and processedRows CSR
    for (int j = 0; j < k - 1; j++) {
        mostCommonNode = (thrust::max_element(dev_ptr, dev_ptr + numNodes) - dev_ptr);
        seeds.insert(mostCommonNode);
        update_counts<<<dimGrid, dimBlock>>>(deviceDataBool, deviceRows, deviceCols, deviceNodeHistogram, numRowsProcessed, numNodes, mostCommonNode);
        cudaDeviceSynchronize();
    }
    mostCommonNode = (thrust::max_element(dev_ptr, dev_ptr + numNodes) - dev_ptr);
    seeds.insert(mostCommonNode);

    cudaFree(deviceDataFloat);
    cudaFree(deviceDataBool);
    cudaFree(deviceRows);
    cudaFree(deviceCols);
    cudaFree(deviceNodeHistogram);
    cudaFree(deviceProcessedRows);
    cudaFree(deviceStates);

    return seeds;
}

int main(int argc, char **argv)
{
    Benchmark b;
    b.setNodeSelectionFunction(nodeSelection);
    b.run();
    return 0;
}
