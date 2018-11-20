#include <algorithm>
#include <sstream>
#include <sys/time.h>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <map>
#include <stdint.h>
#include <math.h>
#include <random>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#define RANDOM_GRAPH_FILEPATH "datasets/random_graph_8000.txt"
#define AUXILIARY_NODE_ID -1
#define L_CONSTANT 1
#define EPSILON_CONSTANT 0.2
#define K_CONSTANT 4
#define BLOCK_SIZE 1024
#define TILE_X_3D 4
#define TILE_Y_3D 16
#define TILE_Z_3D 16
#define TILE_X_2D 32
#define TILE_Y_2D 32

using namespace std;

typedef struct CSR {
    vector<float> data;
    vector<int> rows;
    vector<int> cols;
    CSR() : data(), rows(), cols() {};
} CSR_t;

class CSVReader {
    string fileName;
    string delimeter;

public:
    CSVReader(string filename, string delm = " ") : fileName(filename), delimeter(delm) { }
    vector<vector<string>> getData();
};


unordered_set<int> findKSeeds(CSR *graph, int k);

double nCr(double n, double k);

double calculateLambda(double n, double k, double l, double e);

unordered_set<int> randomReverseReachableSet(CSR *graph);

int width(CSR *graph, unordered_set<int> nodes);

double kptEstimation(CSR *graph, int k);

CSR *covertToCSR(vector<vector<string>> rawData);

bool fileExists (const std::string& name);
