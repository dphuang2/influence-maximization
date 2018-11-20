#include <iostream>
#include <random>
#include <utility>
#include <chrono>
#include <fstream>
#include <unordered_set>
#include <vector>
#include <map>
#include <sys/stat.h>
#include <unistd.h>
#include <iterator>
#include <math.h>
#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>

#define RANDOM_GRAPH_FILEPATH "datasets/random_graph_10000.txt"
#define AUXILIARY_NODE_ID -1
#define L_CONSTANT 1
#define EPSILON_CONSTANT 0.2
#define K_CONSTANT 2
#define BLOCK_SIZE 1024
#define TILE_X 1
#define TILE_Y 32
#define TILE_Z 32

using namespace std;
using namespace std::chrono;

typedef struct CSR {
    vector<double> data;
    vector<int> rows;
    vector<int> cols;
    CSR() : data(), rows(), cols() {};
} CSR_t;

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

pair<int, unordered_set<int>> findMostCommonNode(map<int, unordered_set<int>> R)
{
    map<int, int> counts;
    unordered_set<int>::iterator j;
    map<int, unordered_set<int>> existsInSet;
    map<int, unordered_set<int>>::iterator i;
    int maximum = 0;
    int mostCommonNode = 0;

    for (i = R.begin(); i != R.end(); i++) {
        int setId = i->first;
        unordered_set<int> unordered_set = i->second;
        for (j = unordered_set.begin(); j != unordered_set.end(); j++) {
            existsInSet[*j].insert(setId);
            counts[*j] += 1;
            if (counts[*j] > maximum) {
                mostCommonNode = *j;
                maximum = counts[*j];
            }
        }
    }
    return make_pair(mostCommonNode, existsInSet[mostCommonNode]);
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

    t1 = high_resolution_clock::now();
    for (int i = 0; i < ceil(theta); i++) {
        R[i] = randomReverseReachableSet(graph);
    }
    t2 = high_resolution_clock::now();
    cout << "Generating RR Sets in nodeSelection: " << duration_cast<microseconds>( t2 - t1 ).count() << endl;

    t1 = high_resolution_clock::now();
    for (int j = 0; j < k; j++) {
        t3 = high_resolution_clock::now();
        pair<int, unordered_set<int>> commonNode = findMostCommonNode(R);
        t4 = high_resolution_clock::now();
        cout << "findMostCommonNode: " << duration_cast<microseconds>( t4 - t3 ).count() << endl;
        seeds.insert(commonNode.first);
        t3 = high_resolution_clock::now();
        for (it = commonNode.second.begin(); it != commonNode.second.end(); it++) {
            R.erase(*it);
        }
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
