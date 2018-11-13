import time
from collections import defaultdict

runtimes = defaultdict(list)

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        runtimes[method.__name__ + ':' + str(len(args[0][1]) - 1)].append(te-ts)
        return result

    return timed

def runtimes_to_csv():
    csv = ''
    for method, times in runtimes.items():
        csv += ','.join([method] + [str(time) for time in times]) + '\n'
    return csv
