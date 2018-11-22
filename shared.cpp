#include "shared.h"

double nCr(double n, double k)
{
    double C[(int)n + 1][(int)k + 1];
    int i, j;

    // Caculate value of Binomial Coefficient in bottom up manner
    for (i = 0; i <= n; i++)
    {
        for (j = 0; j <= min((double)i, k); j++)
        {
            // Base Cases
            if (j == 0 || j == i)
                C[i][j] = 1;

            // Calculate value using previosly stored values
            else
                C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
        }
    }

    return C[(int)n][(int)k];
}

double calculateLambda(double n, double k, double l, double e)
{
    return (8.0 + 2 * e) * n * (l * log(n) + log(nCr(n, k)) + log(2) * pow(e, -2));
}

unordered_set<int> randomReverseReachableSet(CSR<float> *graph)
{
    // Seed our randomness
    random_device random_device;
    mt19937 engine{random_device()};

    double n = double(graph->rows.size() - 1);
    uniform_int_distribution<int> dist(0, n - 1);
    int start = dist(engine);
    vector<int> stack{start};
    unordered_set<int> visited;
    while (!stack.empty())
    {
        int currentNode = stack.back();
        stack.pop_back();
        if (visited.count(currentNode) == 0)
        {
            visited.insert(currentNode);
            int dataStart = graph->rows[currentNode];
            int dataEnd = graph->rows[currentNode + 1];

            for (int i = dataStart; i < dataEnd; i++)
            {
                if (((double)rand() / RAND_MAX) < graph->data[i])
                {
                    stack.push_back(graph->cols[i]);
                }
            }
        }
    }
    return visited;
}

int width(CSR<float> *graph, unordered_set<int> nodes)
{
    int count = 0;
    unordered_set<int>::iterator it;
    for (it = nodes.begin(); it != nodes.end(); it++)
    {
        int dataStart = graph->rows[*it];
        int dataEnd = graph->rows[*it + 1];
        count += dataEnd - dataStart;
    }
    return count;
}

double kptEstimation(CSR<float> *graph, int k)
{
    double n = double(graph->rows.size() - 1);
    double m = double(graph->data.size());
    for (int i = 1; i < log2(n); i++)
    {
        double ci = 6 * L_CONSTANT * log(n) + 6 * log(log2(n)) * pow(2, i);
        double sum = 0;
        for (int j = 0; j < ci; j++)
        {
            unordered_set<int> R = randomReverseReachableSet(graph);
            int w_r = width(graph, R);
            double k_r = 1 - pow((1 - (w_r / m)), k);
            sum += k_r;
        }
        if (sum / ci > 1 / pow(2, i))
        {
            return n * sum / (2 * ci);
        }
    }
    return 1.0;
}

bool fileExists(const std::string &name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

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
    while (getline(file, line))
    {
        vector<string> vec;
        istringstream iss(line);
        for (string s; iss >> s;)
            vec.push_back(s);
        dataList.push_back(vec);
    }
    // Close the File
    file.close();

    return dataList;
}

CSR<float> *covertToCSR(vector<vector<string>> rawData)
{
    CSR<float> *graph = new CSR<float>();
    vector<string> data = rawData[0];
    vector<string> rows = rawData[1];
    vector<string> cols = rawData[2];

    for (size_t i = 0; i < data.size(); i++)
    {
        graph->data.push_back(stof(data[i]));
    }
    for (size_t i = 0; i < rows.size(); i++)
    {
        graph->rows.push_back(stoi(rows[i]));
    }
    for (size_t i = 0; i < cols.size(); i++)
    {
        graph->cols.push_back(stoi(cols[i]));
    }

    return graph;
}

Benchmark::Benchmark() {
    files.push_back("datasets/random_graph_20.txt");
    files.push_back("datasets/random_graph_30.txt");
    files.push_back("datasets/random_graph_40.txt");
    files.push_back("datasets/random_graph_50.txt");
    files.push_back("datasets/random_graph_60.txt");
    files.push_back("datasets/random_graph_70.txt");
    files.push_back("datasets/random_graph_80.txt");
    files.push_back("datasets/random_graph_90.txt");
    files.push_back("datasets/random_graph_100.txt");
    files.push_back("datasets/random_graph_800.txt");
    files.push_back("datasets/random_graph_5000.txt");
    files.push_back("datasets/random_graph_8000.txt");
    files.push_back("datasets/random_graph_10000.txt");
    files.push_back("datasets/random_graph_30000.txt");
}

void Benchmark::run() {
    for (int file = 0; file < files.size(); file++) {
        string filepath = files[file];
        if (!fileExists(filepath))
        {
            cout << "File " << filepath << " did not exist...exiting" << endl;
            continue;
        }
        printf("Running trials on %s\n", filepath.c_str());

        // Creating an object of CSVWriter
        CSVReader reader(filepath);
        // Get the data from CSV File
        CSR<float> *graph = covertToCSR(reader.getData());

        struct timeval t1, t2;
        for (int i = 0; i < NUM_TRIALS; i++)
        {
            gettimeofday(&t1, NULL);
            unordered_set<int> seeds = findKSeeds(graph, K_CONSTANT);
            gettimeofday(&t2, NULL);
            unordered_set<int>::iterator it;
            printf("findKSeeds: ");
            for (it = seeds.begin(); it != seeds.end(); it++)
            {
                cout << *it << " ";
            }
            printf("- %ld\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec - t1.tv_usec));
        }
    }
}

void Benchmark::setNodeSelectionFunction(unordered_set<int> (*func)(CSR<float> *graph, int k, double theta)) {
    nodeSelection = func;
}

unordered_set<int> Benchmark::findKSeeds(CSR<float> *graph, int k)
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
    printf("nodeSelection: %ld\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec - t1.tv_usec));

    return selectedNodes;
}
