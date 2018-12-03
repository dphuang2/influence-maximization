import sys
from collections import defaultdict

def convert_to_csv(filename, num_trials):
    csv = 'num_nodes'

    with open(filename, 'r') as fp:
        lines = fp.read().split('\n')

    for i in range(num_trials):
        csv += ',trial {}'.format(i)
    csv += '\n'

    data = defaultdict(list) # Map from number of nodes to time
    current_num_nodes = -1
    for line in lines:
        if 'Trial' in line:
            current_num_nodes = int(line.split('_')[-1].split('.')[0])
        elif 'findKSeeds' in line:
            data[current_num_nodes].append(float(line.split(' ')[-1]) * 1e-6)

    for num_nodes in sorted(data.keys()):
        csv += str(num_nodes)
        for trial in data[num_nodes]:
            csv += ',{}'.format(trial)
        csv += '\n'

    return csv

if __name__=="__main__":
    if len(sys.argv) != 3:
        print 'Pass in path of .log file and # of trials'
        exit()
    filepath = sys.argv[1]
    num_nodes = int(sys.argv[2])
    with open(filepath.split('.')[0] + '.csv', 'w') as fp:
        fp.write(convert_to_csv(filepath, num_nodes))
