#include "shared.h"
#include <curand_kernel.h>

using namespace std;

typedef struct Node
{
    int id;
    Node *prev;
    Node *next;
    __device__ Node(int id) : id(id){};
} Node_t;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int maxIndex(int * array, int size) {
    int max = 0;
    for (int i = 0; i < size; i++) {
        if (array[i] > array[max])
            max = i;
    }
    return max;
}

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
        Node *stack = new Node(randomNodeId);
        Node *auxiliary = new Node(AUXILIARY_NODE_ID);
        auxiliary->next = stack;
        stack->prev = auxiliary;

        // Returns false when stack is NULL
        while (stack != NULL && stack->id != AUXILIARY_NODE_ID)
        {
            // pop from stack
            int currentNodeId = stack->id;
            Node *temp = stack;
            stack = stack->prev;
            delete temp;

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
                        stack->next = new Node(cols[i]);
                        stack->next->prev = stack;
                        stack = stack->next;
                    }
                }
            }
        }
        delete auxiliary;
    }
}

__global__ void update_counts(bool *data, int *rows, int *cols, int *histogram, int numRows, int numNodes, int nodeToDelete)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= numRows)
        return;

    int dataStart = rows[row];
    int dataEnd = rows[row + 1];

    bool exists = false;
    // Lets first figure out if this set includes the delete node and still exists
    for (int i = dataStart; i < dataEnd; i++)
    {
        if (data[i] && cols[i] == nodeToDelete)
        {
            exists = true;
            break;
        }
    }

    if (exists)
    {
        // Now lets set all of them to false and update histogram concurrently
        for (int i = dataStart; i < dataEnd; i++)
        {
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
    float *deviceDataFloat;
    bool *deviceDataBool;
    int *deviceRows;
    int *deviceCols;
    int *deviceNodeHistogram;
    int *hostNodeHistogram;
    bool *deviceProcessedRows;
    bool *hostProcessedRows;
    curandState *deviceStates;

    // Initialize data, rows, and cols
    int sizeOfData = graph->data.size() * sizeof(float);
    int sizeOfRows = graph->rows.size() * sizeof(int);
    int sizeOfCols = graph->cols.size() * sizeof(int);
    CUDA_CHECK(cudaMalloc((void **)&deviceDataFloat, sizeOfData));
    CUDA_CHECK(cudaMalloc((void **)&deviceRows, sizeOfRows));
    CUDA_CHECK(cudaMalloc((void **)&deviceCols, sizeOfCols));
    CUDA_CHECK(cudaMemcpy(deviceDataFloat, &(graph->data[0]), sizeOfData, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceRows, &(graph->rows[0]), sizeOfRows, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceCols, &(graph->cols[0]), sizeOfCols, cudaMemcpyHostToDevice));

    // Initialize output of kernel
    int numNodes = (int)graph->rows.size() - 1;
    int sizeOfNodeHistogram = numNodes * sizeof(int);
    CUDA_CHECK(cudaMalloc((void **)&deviceNodeHistogram, sizeOfNodeHistogram));
    CUDA_CHECK(cudaMemset(deviceNodeHistogram, 0, sizeOfNodeHistogram));
    hostNodeHistogram = (int*) malloc(sizeOfNodeHistogram);

    // Calculate number of batches
    int numBatches = ceil(theta / NUM_ROWS_PER_BATCH);

    // Initialize processed rows output
    long long int sizeOfProcessedRows = sizeof(bool) * NUM_ROWS_PER_BATCH * numNodes;
    CUDA_CHECK(cudaMalloc((void **)&deviceProcessedRows, sizeOfProcessedRows));
    hostProcessedRows = (bool *)malloc(sizeOfProcessedRows);

    // Initialize RNG States
    CUDA_CHECK(cudaMalloc((void **)&deviceStates, NUM_ROWS_PER_BATCH * sizeof(curandState)));
    dim3 dimGrid((NUM_ROWS_PER_BATCH / BLOCK_SIZE) + 1, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    init_rng<<<dimGrid, dimBlock>>>(NUM_ROWS_PER_BATCH, deviceStates, 1, 0);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Process batches
    int numRowsProcessed = 0;
    // Can't use bool: https://stackoverflow.com/questions/8399417/why-vectorboolreference-doesnt-return-reference-to-bool
    CSR<char> *processedRows = new CSR<char>();
    for (int i = 0; i < numBatches; i++)
    {
        // Reset processed rows for new kernel
        CUDA_CHECK(cudaMemset(deviceProcessedRows, false, sizeOfProcessedRows));

        // Process the minimum number of rows
        int numRowsToProcess = min(NUM_ROWS_PER_BATCH, (int)ceil(theta) - numRowsProcessed);

        // Launch RR generation kernel
        dimGrid = dim3(ceil(float(numRowsToProcess) / BLOCK_SIZE), 1, 1);
        dimBlock = dim3(BLOCK_SIZE, 1, 1);
        generate_rr_sets<<<dimGrid, dimBlock>>>(deviceDataFloat, deviceRows, deviceCols, deviceProcessedRows, deviceNodeHistogram, numNodes, numRowsToProcess, deviceStates);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Add to our running processedRows CSR using the rows and cols members
        CUDA_CHECK(cudaMemcpy(hostProcessedRows, deviceProcessedRows, sizeOfProcessedRows, cudaMemcpyDeviceToHost));
        for (int j = 0; j < numRowsToProcess; j++)
        {
            processedRows->rows.push_back(processedRows->data.size());
            for (int k = 0; k < numNodes; k++)
            {
                if (hostProcessedRows[(unsigned long long int)j * numNodes + k])
                {
                    processedRows->data.push_back(true);
                    processedRows->cols.push_back(k);
                }
            }
        }
        processedRows->rows.push_back(processedRows->data.size());

        numRowsProcessed += numRowsToProcess;
    }

    // Copy our processedRows CSR to device
    CUDA_CHECK(cudaFree(deviceRows));
    CUDA_CHECK(cudaFree(deviceCols));
    CUDA_CHECK(cudaFree(deviceDataFloat));
    sizeOfData = processedRows->data.size() * sizeof(char);
    sizeOfRows = processedRows->rows.size() * sizeof(int);
    sizeOfCols = processedRows->cols.size() * sizeof(int);
    CUDA_CHECK(cudaMalloc((void **)&deviceDataBool, sizeOfData));
    CUDA_CHECK(cudaMalloc((void **)&deviceRows, sizeOfRows));
    CUDA_CHECK(cudaMalloc((void **)&deviceCols, sizeOfCols));
    CUDA_CHECK(cudaMemcpy(deviceDataBool, &(processedRows->data[0]), sizeOfData, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceRows, &(processedRows->rows[0]), sizeOfRows, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceCols, &(processedRows->cols[0]), sizeOfCols, cudaMemcpyHostToDevice));
    
    // Initialize dimensions for count updating
    unsigned int mostCommonNode;
    dimGrid = dim3(ceil(float(numRowsProcessed) / BLOCK_SIZE), 1, 1);
    dimBlock = dim3(BLOCK_SIZE, 1, 1);

    // Copy out node histogram to host
    CUDA_CHECK(cudaMemcpy(hostNodeHistogram, deviceNodeHistogram, sizeOfNodeHistogram, cudaMemcpyDeviceToHost));

    // Select nodes using histogram and processedRows CSR
    for (int j = 0; j < k - 1; j++)
    {
        mostCommonNode = maxIndex(hostNodeHistogram, numNodes);
        seeds.insert(mostCommonNode);
        update_counts<<<dimGrid, dimBlock>>>(deviceDataBool, deviceRows, deviceCols, deviceNodeHistogram, numRowsProcessed, numNodes, mostCommonNode);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hostNodeHistogram, deviceNodeHistogram, sizeOfNodeHistogram, cudaMemcpyDeviceToHost));
    }
    mostCommonNode = maxIndex(hostNodeHistogram, numNodes);
    seeds.insert(mostCommonNode);

    CUDA_CHECK(cudaFree(deviceDataBool));
    CUDA_CHECK(cudaFree(deviceRows));
    CUDA_CHECK(cudaFree(deviceCols));
    CUDA_CHECK(cudaFree(deviceProcessedRows));
    CUDA_CHECK(cudaFree(deviceStates));
    free(hostProcessedRows);
    delete processedRows;

    return seeds;
}

int main(int argc, char **argv)
{
    Benchmark b;
    b.setNodeSelectionFunction(nodeSelection);
    b.run();
    return 0;
}
