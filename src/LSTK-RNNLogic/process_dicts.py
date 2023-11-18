import sys

dataset = sys.argv[1]
data_path = './data'
idx = 0
fw = open('{}/{}/entities.dict'.format(data_path, dataset), mode='w')
with open('{}/{}/entities.txt'.format(data_path, dataset), mode='r') as fd:
    for line in fd:
        if not line: continue
        fw.write('{}\t{}\n'.format(idx, line.strip()))
        idx += 1
fw.close()

idx = 0
fw = open('{}/{}/relations.dict'.format(data_path, dataset), mode='w')
with open('{}/{}/relations_ext.txt'.format(data_path, dataset), mode='r') as fd:
    for line in fd:
        if not line: continue
        fw.write('{}\t{}\n'.format(idx, line.strip()))
        idx += 1
fw.close()