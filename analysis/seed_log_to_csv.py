import sys
import pdb
from collections import defaultdict

def convert_to_csv(filename, num_trials):
    csv = 'dataset'

    with open(filename, 'r') as fp:
        lines = fp.read().split('\n')

    for i in range(1, num_trials + 1):
        csv += ',seeds {}'.format(i)
    csv += '\n'

    data = defaultdict(lambda: defaultdict(list)) # Map from number of nodes to time
    for line in lines:
        if 'findKSeeds' in line:
            seconds = float(line.split(' ')[-1]) * 1e-6
            num_seeds = int(line.split('(')[1].split(')')[0])
            filename = line.split('[')[1].split(']')[0]
            data[filename][num_seeds].append(seconds)

    for dataset in sorted(data.keys()):
        csv += dataset
        for k in sorted(data[dataset].keys()):
            average = sum(data[dataset][k]) / len(data[dataset][k])
            pdb.set_trace()
            csv += ',{}'.format(average)
        csv += '\n'

    return csv

if __name__=="__main__":
    if len(sys.argv) != 3:
        print('Pass in path of .log file and # of seeds tried')
        exit()
    filepath = sys.argv[1]
    num_trials = int(sys.argv[2])
    with open(filepath.split('.')[0] + '.csv', 'w') as fp:
        fp.write(convert_to_csv(filepath, num_trials))
