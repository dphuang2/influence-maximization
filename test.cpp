#include <iostream>

using namespace std;

typedef struct node
{ 
    int id;
    node *prev;
    node *next;
    node(int id) : id(id){};
} node;

int main() {
    node * test = new node(1);
    cout << test->id << endl;
    free(test);
    return 1;
}
