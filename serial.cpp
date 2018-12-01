#include "shared.h"

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

pair<unordered_set<int>, int> nodeSelection(CSR<float> *graph, int k, double theta)
{
    unordered_set<int>::iterator it;
    unordered_set<int> seeds;
    map<int, unordered_set<int>> R;

    for (int i = 0; i < ceil(theta); i++)
    {
        R[i] = randomReverseReachableSet(graph);
    }

    for (int j = 0; j < k; j++)
    {
        pair<int, unordered_set<int>> commonNode = findMostCommonNode(R);
        seeds.insert(commonNode.first);
        for (it = commonNode.second.begin(); it != commonNode.second.end(); it++)
        {
            R.erase(*it);
        }
    }

    return make_pair(seeds, R.size());
}

int main(int argc, char **argv)
{
    Benchmark b;
    b.setNodeSelectionFunction(nodeSelection);
    b.run();
    return 0;
}
