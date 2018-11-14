import time
from collections import defaultdict

runtimes = defaultdict(list)
find_k_seeds_runtimes = defaultdict(list)
cumulative_runtimes = defaultdict(float)
execution_counts = defaultdict(int)

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        runtimes[method.__name__].append(te-ts)
        execution_counts[method.__name__] += 1
        cumulative_runtimes[method.__name__] += te - ts
        if method.__name__ == 'find_k_seeds':
            find_k_seeds_runtimes[method.__name__ + ':' + str(len(args[0][1]) - 1)].append(te-ts)
        return result

    return timed

def to_csv(dictionary):
    csv = ''
    for method, value in dictionary.items():
        if type(value) == list:
            csv += ','.join([method] + [str(value) for value in value]) + '\n'
        else:
            csv += ','.join([method, str(value)]) + '\n'
    return csv