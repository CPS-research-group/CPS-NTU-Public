import re
import sys


times = []
with open(f'{sys.argv[1]}.txt', 'r') as f:
    for line in f:
        m = re.search(r'\((?P<inference>\d+\.\d+)ms\)\s+Inference,\s+\((?P<nms>\d+\.\d+)ms\)\s+NMS', line.strip())
        print(m)
        if m:
            t = m.groupdict()
            times.append(float(t['inference']) + float(t['nms']))
with open(f'processed_times{sys.argv[1]}.txt', 'w') as g:
    g.write(','.join([str(i) for i in times])) 
	
