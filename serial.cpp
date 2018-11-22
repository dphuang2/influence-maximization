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

    return seeds;
}

int main(int argc, char **argv)
{
    Benchmark b;
    b.setNodeSelectionFunction(nodeSelection);
    b.run();
    return 0;
}
