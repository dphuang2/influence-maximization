#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
#include <curand_kernel.h>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <math.h>
#include <random>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#define RANDOM_GRAPH_FILEPATH "datasets/random_graph_100.txt"
#define AUXILIARY_NODE_ID -1
#define L_CONSTANT 1
#define EPSILON_CONSTANT 0.2
#define K_CONSTANT 2
#define BLOCK_SIZE 1024
#define TILE_X_3D 1
#define TILE_Y_3D 32
#define TILE_Z_3D 32
#define TILE_X_2D 32
#define TILE_Y_2D 32

using namespace std;
using namespace std::chrono;

typedef struct CSR {
    vector<double> data;
    vector<int> rows;
    vector<int> cols;
    CSR() : data(), rows(), cols() {};
} CSR_t;

typedef struct node {
    int id;
    node *prev;
    node *next;
    __device__ node(int id) : id(id) {};
} node_t;

__global__ void init_rng(int nthreads, curandState *states, unsigned long long seed, unsigned long long offset)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nthreads)
        return;
    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(seed, id, offset, &states[id]);
}

__global__ void generate_rr_sets(float *data, int *rows, int *cols, bool *out, int *nodeHistogram, int numNodes, int numSets, curandState *states)
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numSets) {
        curandState *state = &states[tid];

        /* Because C does not give us the luxury of dynamic arrays, to imitate the
        behavior of a stack, I am using a linked list*/
        int randomNodeId = ceil(numNodes * curand_uniform(state)) - 1;
        node *stack = new node(randomNodeId);
        node *auxiliary = new node(AUXILIARY_NODE_ID);
        auxiliary->next = stack;
        stack->prev = auxiliary;

        // Returns false when stack is NULL
        while (stack->id != AUXILIARY_NODE_ID) {
            // pop from stack
            int currentNodeId = stack->id;
            node *temp = stack;
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
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int node_y = blockDim.y * blockIdx.y + threadIdx.y;
    int node_z = blockDim.z * blockIdx.z + threadIdx.z;

    if (row < num_rows && node_y < num_nodes && node_z < num_nodes) {
        if (batch[row * num_nodes + node_y] && batch[row * num_nodes + node_z]) {
            atomicAdd(&counts[node_y * num_nodes + node_z], 1);
        }
    }
}

__global__ void update_counts(int * intersections, int * histogram, int numNodes, int nodeToDelete)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= numNodes || col >= numNodes)
        return;
    if (row >= col) {
        intersections[row * numNodes + col] -= intersections[nodeToDelete * numNodes + col];
        if (row == col) {
            histogram[row] = intersections[row * numNodes + col];
        }
    }
    else if (row < col) {
        intersections[row * numNodes + col] -= intersections[row * numNodes + nodeToDelete];
    }
}

bool fileExists (const std::string& name)
{
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

/*
 * A class to read data from a csv file.
 */
class CSVReader {
    string fileName;
    string delimeter;

public:
    CSVReader(string filename, string delm = " ") : fileName(filename), delimeter(delm)
    {
    }

    // Function to fetch data from a CSV File
    vector<vector<string>> getData();
};

/*
 * Parses through csv file line by line and returns the data
 * in vector of vector of strings.
 */
vector<vector<string>> CSVReader::getData()
{
    ifstream file(fileName);

    vector<vector<string>> dataList;

    string line = "";
    // Iterate through each line and split the content using delimeter
    while (getline(file, line)) {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        dataList.push_back(vec);
    }
    // Close the File
    file.close();

    return dataList;
}

CSR *covertToCSR(vector<vector<string>> rawData)
{
    CSR *graph = new CSR();
    vector<string> data = rawData[0];
    vector<string> rows = rawData[1];
    vector<string> cols = rawData[2];

    for (size_t i = 0; i < data.size(); i++) {
        graph->data.push_back(stof(data[i]));
    }
    for (size_t i = 0; i < rows.size(); i++) {
        graph->rows.push_back(stoi(rows[i]));
    }
    for (size_t i = 0; i < cols.size(); i++) {
        graph->cols.push_back(stoi(cols[i]));
    }

    return graph;
}

int nCr(int n, int k)
{
    int C[n + 1][k + 1];
    int i, j;

    // Caculate value of Binomial Coefficient in bottom up manner
    for (i = 0; i <= n; i++) {
        for (j = 0; j <= min(i, k); j++) {
            // Base Cases
            if (j == 0 || j == i)
                C[i][j] = 1;

            // Calculate value using previosly stored values
            else
                C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
        }
    }

    return C[n][k];
}

double calculateLambda(double n, int k, double l, double e)
{
    return (double)(8.0 + 2 * e) * n * (l * log(n) + log(nCr(n, k)) + log(2) * pow(e, -2));
}

unordered_set<int> randomReverseReachableSet(CSR *graph)
{
    // Seed our randomness
    random_device random_device;
    mt19937 engine{random_device()};

    double n = double(graph->rows.size() - 1);
    uniform_int_distribution<int> dist(0, n - 1);
    int start = dist(engine);
    vector<int> stack{start};
    unordered_set<int> visited;
    while (!stack.empty()) {
        int currentNode = stack.back();
        stack.pop_back();
        if (visited.count(currentNode) == 0) {
            visited.insert(currentNode);
            int dataStart = graph->rows[currentNode];
            int dataEnd = graph->rows[currentNode + 1];

            for (int i = dataStart; i < dataEnd; i++) {
                if (((double)rand() / RAND_MAX) < graph->data[i]) {
                    stack.push_back(graph->cols[i]);
                }
            }
        }
    }
    return visited;
}

int width(CSR *graph, unordered_set<int> nodes)
{
    int count = 0;
    unordered_set<int>::iterator it;
    for (it = nodes.begin(); it != nodes.end(); it++) {
        int dataStart = graph->rows[*it];
        int dataEnd = graph->rows[*it + 1];
        count += dataEnd - dataStart;
    }
    return count;
}

double kptEstimation(CSR *graph, int k)
{
    double n = double(graph->rows.size() - 1);
    double m = double(graph->data.size());
    for (int i = 1; i < log2(n); i++) {
        double ci = 6 * L_CONSTANT * log(n) + 6 * log(log2(n)) * pow(2, i);
        double sum = 0;
        for (int j = 0; j < ci; j++) {
            unordered_set<int> R = randomReverseReachableSet(graph);
            int w_r = width(graph, R);
            double k_r = 1 - pow((1 - (w_r / m)), k);
            sum += k_r;
        }
        if (sum / ci > 1 / pow(2, i)) {
            return n * sum / (2 * ci);
        }
    }
    return 1.0;
}

unordered_set<int> nodeSelection(CSR *graph, int k, double theta)
{
    unordered_set<int>::iterator it;
    unordered_set<int> seeds;
    map<int, unordered_set<int>> R;
    high_resolution_clock::time_point t1;
    high_resolution_clock::time_point t2;
    high_resolution_clock::time_point t3;
    high_resolution_clock::time_point t4;
    size_t * freeGPUBytes;
    size_t * totalGPUBytes;
    float * deviceData;
    int * deviceRows;
    int * deviceCols;
    int * deviceNodeHistogram;
    int * deviceNodeToNodeIntersections;
    bool * deviceProcessedRows;
    curandState * deviceStates;

    // Initialize data, rows, and cols
    int sizeOfData = graph->data.size() * sizeof(float);
    int sizeOfRows = graph->rows.size() * sizeof(int);
    int sizeOfCols = graph->cols.size() * sizeof(int);
    cudaMalloc((void **) &deviceData, sizeOfData);
    cudaMalloc((void **) &deviceRows, sizeOfRows);
    cudaMalloc((void **) &deviceCols, sizeOfCols);
    cudaMemcpy(deviceData, &(graph->data[0]), sizeOfData, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceRows, &(graph->rows[0]), sizeOfRows, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceCols, &(graph->cols[0]), sizeOfCols, cudaMemcpyHostToDevice);

    // Initialize output of kernel
    int numNodes = (int) graph->rows.size() - 1;
    int sizeOfNodeHistogram = sizeof(int) * numNodes;
    int sizeOfNodeToNodeIntersections = sizeof(int) * numNodes * numNodes;
    cudaMalloc((void **) &deviceNodeHistogram, sizeOfNodeHistogram);
    cudaMalloc((void **) &deviceNodeToNodeIntersections, sizeOfNodeToNodeIntersections);
    cudaMemset(deviceNodeHistogram, 0, sizeOfNodeHistogram);
    cudaMemset(deviceNodeToNodeIntersections, 0, sizeOfNodeToNodeIntersections);

    // Calculate number of batches
    cudaMemGetInfo(freeGPUBytes, totalGPUBytes);
    double numRowsPerBatch = ceil(
                                 ((*freeGPUBytes / 1.5) - (4 * numNodes + pow(numNodes, 2)))
                                 /
                                 (numNodes + sizeof(curandState)));
    int numBatches = ceil(theta / numRowsPerBatch);
    int sizeOfProcessedRows = sizeof(bool) * numRowsPerBatch * numNodes;

    // Initialize processed rows output
    cudaMalloc((void **) &deviceProcessedRows, sizeOfProcessedRows);

    // Initialize RNG States
    cudaMalloc((void **) &deviceStates, numRowsPerBatch * sizeof(curandState));
    dim3 dimGrid(ceil(float(numRowsPerBatch) / BLOCK_SIZE), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    init_rng<<<dimGrid, dimBlock>>>(numRowsPerBatch, deviceStates, 1, 0);

    // Process batches
    t1 = high_resolution_clock::now();
    int numRowsProcessed = 0;
    for (int i = 0; i < numBatches; i++) {
        cudaMemset(deviceProcessedRows, false, sizeOfProcessedRows);
        int numRowsToProcess = min(numRowsPerBatch, theta - numRowsProcessed);
        dimGrid = dim3(ceil(float(numRowsToProcess) / BLOCK_SIZE), 1, 1);
        dimBlock = dim3(BLOCK_SIZE, 1, 1);
        generate_rr_sets<<<dimGrid, dimBlock>>>(deviceData, deviceRows, deviceCols, deviceProcessedRows, deviceNodeHistogram, numNodes, numRowsToProcess, deviceStates);

        dimGrid = dim3(ceil(float(numRowsToProcess) / TILE_X_3D), ceil(float(numNodes / TILE_Y_3D)), ceil(float(numNodes) / TILE_Z_3D));
        dimBlock = dim3(TILE_X_3D, TILE_Y_3D, TILE_Z_3D);
        count_node_to_node_intersections<<<dimGrid, dimBlock>>>(deviceNodeToNodeIntersections, deviceProcessedRows, numRowsToProcess, numNodes);
    }
    t2 = high_resolution_clock::now();
    cout << "Generating RR Sets in nodeSelection: " << duration_cast<microseconds>( t2 - t1 ).count() << endl;

    t1 = high_resolution_clock::now();
    thrust::device_ptr<int> dev_ptr(deviceNodeHistogram);
    for (int j = 0; j < k; j++) {
        t3 = high_resolution_clock::now();
        int mostCommonNode = *thrust::max_element(dev_ptr, dev_ptr + numNodes);
        t4 = high_resolution_clock::now();
        cout << "findMostCommonNode: " << duration_cast<microseconds>( t4 - t3 ).count() << endl;
        seeds.insert(mostCommonNode);
        t3 = high_resolution_clock::now();
        dimGrid = dim3(ceil(float(numNodes) / TILE_X_2D), ceil(float(numNodes) / TILE_Y_2D), 1);
        dimBlock = dim3(TILE_X_2D, TILE_Y_2D, 1);
        update_counts<<<dimGrid, dimBlock>>>(deviceNodeToNodeIntersections, deviceNodeHistogram, numNodes, mostCommonNode);
        t4 = high_resolution_clock::now();
        cout << "deleting sets the most commond node exists in : " << duration_cast<microseconds>( t4 - t3 ).count() << endl;
    }
    t2 = high_resolution_clock::now();
    cout << "Selecting seeds: " << duration_cast<microseconds>( t2 - t1 ).count() << endl;

    return seeds;
}

unordered_set<int> findKSeeds(CSR *graph, int k)
{
    double n = double(graph->rows.size() - 1);
    high_resolution_clock::time_point t1;
    high_resolution_clock::time_point t2;

    t1 = high_resolution_clock::now();
    double kpt = kptEstimation(graph, k);
    t2 = high_resolution_clock::now();
    cout << "kptEstimation: " <<  duration_cast<microseconds>( t2 - t1 ).count() << endl;

    t1 = high_resolution_clock::now();
    double lambda = calculateLambda(n, k, L_CONSTANT, EPSILON_CONSTANT);
    t2 = high_resolution_clock::now();
    cout << "calculateLambda: " <<  duration_cast<microseconds>( t2 - t1 ).count() << endl;

    double theta = lambda / kpt;

    t1 = high_resolution_clock::now();
    unordered_set<int> selectedNodes = nodeSelection(graph, k, theta);
    t2 = high_resolution_clock::now();
    cout << "selectedNodes: " <<  duration_cast<microseconds>( t2 - t1 ).count() << endl;

    return selectedNodes;
}

int main(int argc, char **argv)
{
    if (!fileExists(RANDOM_GRAPH_FILEPATH)) {
        cout << "File " << RANDOM_GRAPH_FILEPATH << " did not exist...exiting" <<endl;
        exit(1);
    }

    // Creating an object of CSVWriter
    CSVReader reader(RANDOM_GRAPH_FILEPATH);
    // Get the data from CSV File
    CSR *graph = covertToCSR(reader.getData());

    for (int i = 0; i < 1; i++) {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        unordered_set<int> seeds = findKSeeds(graph, K_CONSTANT);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>( t2 - t1 ).count();
        unordered_set<int>::iterator it;
        for (it = seeds.begin(); it != seeds.end(); it++) {
            cout << *it << " ";
        }
        cout << "- " << duration << " microseconds" << endl;
    }
    return 0;
}
