import sys

dataset = sys.argv[1]
data_path = './data'
for split in ['train', 'valid', 'test']:
    fact_path = '{}/{}/{}_triple_scores.txt'.format(data_path, dataset, split)
    tuples = set()
    with open('{}/{}/{}.txt'.format(data_path, dataset, split), mode='r') as fd:
        for line in fd:
            if not line: continue
            h, r, t = line.strip().split('\t')
            tuples.add('{}||{}||{}'.format(h, r, t).replace(' ', '_'))
    fw = open('{}/{}/{}_ext.txt'.format(data_path, dataset, split), mode='w')
    with open(fact_path, mode='r') as fd:
        for i, line in enumerate(fd.readlines()):
            if not line: continue
            items = line.strip().split('\t')
            if len(items) != 3: continue
            triple, score, pred = items
            score = float(score)
            if score < 0.5: continue
            triple = triple.replace(' ', '_')
            h, r, t = triple.split('||')
            tuple = '{}||{}||{}'.format(h, r, t)
            # if tuple in tuples: continue
            fw.write('{}\t{}\t{}\n'.format(h, r + '_ext', t))
    fw.close()
