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

double calculateLambdaPrime(int n, int k, double l, double eps) 
{
    return (2.0 + 2.0/3.0 * eps) * (log(nCr(n, k)) + l * log(n) * log(log2(n))) * n * pow(eps, -2);
}

double calculateLambdaStar(int n, int k, double l, double eps) 
{

    double alpha = sqrt(l * log(n) + log(2));
    double beta = sqrt((1 - (1 / M_E)) * (log(nCr(n, k)) + l * log(n) + log(2)));
    return 2 * n * pow(((1 - (1 / M_E)) * alpha + beta), 2) * pow(eps,-2);
}

double Benchmark::findTheta(CSR<float> *graph, int n, int k, double e, double l)
{
    double lb = 1;
    double eps = e * sqrt(2);
    double lam = calculateLambdaPrime(n, k, l, eps);
    double lamStar = calculateLambdaStar(n, k, l, e);

    for (int i = 0; i < ceil(log2(n)) - 1; i++)
    {
        double x = n / pow(2.0, i + 1);
        double theta = lam / x;
        int uncovered = nodeSelection(graph, k, theta).second;
        double frac = (theta - uncovered) / theta;
        if (n * frac >= (1 + eps) * x)
            return lamStar / (n * frac / (1 + eps));
    }
    return lamStar;
}

bool fileExists(const std::string &name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

long int calcTimeDiff(struct timeval t1, struct timeval t2)
{
    return ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec - t1.tv_usec);
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

void Benchmark::run(int k)
{
    setbuf(stdout, NULL);
    for (int file = 0; file < files.size(); file++)
    {
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
            printf("Trial %d - %s\n", i, filepath.c_str());
            gettimeofday(&t1, NULL);
            unordered_set<int> seeds = findKSeeds(graph, k);
            gettimeofday(&t2, NULL);
            unordered_set<int>::iterator it;
            printf("findKSeeds(%d)[%s]: ", k, filepath.c_str());
            for (it = seeds.begin(); it != seeds.end(); it++)
            {
                cout << *it << " ";
            }
            printf("- %ld\n", calcTimeDiff(t1, t2));
        }
        delete graph;
    }
}

void Benchmark::runMany(int k) 
{
    for (int i = 1; i <= k; i++)
        run(i);
}



void Benchmark::setNodeSelectionFunction(pair<unordered_set<int>, int> (*func)(CSR<float> *graph, int k, double theta))
{
    nodeSelection = func;
}

unordered_set<int> Benchmark::findKSeeds(CSR<float> *graph, int k)
{
    double n = double(graph->rows.size() - 1);
    struct timeval t1;
    struct timeval t2;

    gettimeofday(&t1, NULL);
    double theta = findTheta(graph, n, k, EPSILON_CONSTANT, L_CONSTANT);
    gettimeofday(&t2, NULL);
    printf("findTheta = %f: %ld\n", theta, calcTimeDiff(t1, t2));

    gettimeofday(&t1, NULL);
    unordered_set<int> selectedNodes = nodeSelection(graph, k, theta).first;
    gettimeofday(&t2, NULL);
    printf("nodeSelection: %ld\n", calcTimeDiff(t1, t2));

    return selectedNodes;
}

Benchmark::Benchmark()
{
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
