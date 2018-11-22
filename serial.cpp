#include "shared.h"

pair<int, unordered_set<int>> findMostCommonNode(map<int, unordered_set<int>> R)
{
    map<int, int> counts;
    unordered_set<int>::iterator j;
    map<int, unordered_set<int>> existsInSet;
    map<int, unordered_set<int>>::iterator i;
    int maximum = 0;
    int mostCommonNode = 0;

    for (i = R.begin(); i != R.end(); i++)
    {
        int setId = i->first;
        unordered_set<int> unordered_set = i->second;
        for (j = unordered_set.begin(); j != unordered_set.end(); j++)
        {
            existsInSet[*j].insert(setId);
            counts[*j] += 1;
            if (counts[*j] > maximum)
            {
                mostCommonNode = *j;
                maximum = counts[*j];
            }
        }
    }
    return make_pair(mostCommonNode, existsInSet[mostCommonNode]);
}

unordered_set<int> nodeSelection(CSR<float> *graph, int k, double theta)
{
    unordered_set<int>::iterator it;
    unordered_set<int> seeds;
    map<int, unordered_set<int>> R;
    struct timeval t1;
    struct timeval t2;
    struct timeval t3;
    struct timeval t4;

    gettimeofday(&t1, NULL);
    for (int i = 0; i < ceil(theta); i++)
    {
        R[i] = randomReverseReachableSet(graph);
    }
    gettimeofday(&t2, NULL);
    printf("Generating RR Sets in nodeSelection: %ld\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec - t1.tv_usec));

    gettimeofday(&t1, NULL);
    for (int j = 0; j < k; j++)
    {
        gettimeofday(&t3, NULL);
        pair<int, unordered_set<int>> commonNode = findMostCommonNode(R);
        gettimeofday(&t4, NULL);
        printf("findMostCommonNode: %ld\n", ((t4.tv_sec - t3.tv_sec) * 1000000L + t4.tv_usec - t3.tv_usec));
        seeds.insert(commonNode.first);
        gettimeofday(&t3, NULL);
        for (it = commonNode.second.begin(); it != commonNode.second.end(); it++)
        {
            R.erase(*it);
        }
        gettimeofday(&t4, NULL);
        printf("deleting sets with the most common node: %ld\n", ((t4.tv_sec - t3.tv_sec) * 1000000L + t4.tv_usec - t3.tv_usec));
    }
    gettimeofday(&t2, NULL);
    printf("Selecting seeds: %ld\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec - t1.tv_usec));

    return seeds;
}

unordered_set<int> findKSeeds(CSR<float> *graph, int k)
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
    CSR<float> *graph = covertToCSR(reader.getData());

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
