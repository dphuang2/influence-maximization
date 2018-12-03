# Influence Maximization
by Tim Murray and Dylan Huang

# Parallel algorithm
## Opportunity
We noticed that almost all of the computation time was dedicated to generating
a lot of reverse reachable sets. We saw this as an opportunity to parallelize
the generation of many RR sets at once. The following algorithm is the
pseudocode for serial node selection. 

```python
Initialize an RR set R
Initialize a node set S
Generate theta RR sets and insert into R
for j = 1 to k
    Identify the node v that covers the most RR sets in R
    Add v into S
    Remove R from all RR sets that are covered by v
end
return S
```

We can see that the generation of RR sets is a single line that can be
parallelized to be O(1) rather than O(theta).

## Algorithm
Our parallel algorithm can be broken into 4 ideas: RR set generation,
node histogram, max reduction, updating node histogram.

### RR set generation
We represent our set of RR sets, R, as a (theta x n) matrix of booleans. We
launch a kernel on our GPU with theta threads and perform BFS and set a node to
true if it had been visited in the search. To set a node to true, we simply
index into R with R[Thread, Node] and set the value to true. We represent our
network as a CSR so we can simply copy blocks of data to the GPU.

### Node histogram
Along with R, we introduce a node histogram called H. The node histogram is a
histogram of node appearances during the RR set generation. We will use this
histogram the parallelize the identification of "node v that covers the most RR
sets in R". To initialize a histogram, we allocate an array of size n in GPU
memory with all 0s. During the RR set generation, we atomically (this is
important since every thread will be updating this histogram) increment a count
in our node histogram for each visited node.

### Max reduction
At the end of our generation, we have two data
structures—our matrix of boolean R, and our list of integers H. For each
iteration of k, we can perform a max reduction and identify our node v in log(n)
time. We can then add v into our seed set.

### Updating node histogram
To update the node histogram, we launch another kernel that will update our node
histogram in parallel. This kernel launches with theta threads and first
discovers if v is in the set.  If it is, it iterates through all nodes in the RR
set and atomically decrements node frequencies. We iterate this k times and
arrive at our final seed set S.

### Pseudocode
```python
Initialize a RR set matrix R
Initialize a node set S
Initialize a node histogram H
Parallel for generate theta RR sets and insert into R and update H # O(1)
for j = 1 to k
    Identify the node v with max reduction on H # O(logn)
    Add v in to S
    Parallel for over theta sets to update node histogram # O(1)
end
return S
```
