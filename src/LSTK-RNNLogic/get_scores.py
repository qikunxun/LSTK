import sys

dataset = sys.argv[1]
data_path = './data'
data = []
for prefix in ['train', 'valid', 'test']:
    fact_path = '{}/{}/{}_triple_scores.txt'.format(data_path, dataset, prefix)
    tuples = set()
    with open('{}/{}/{}.txt'.format(data_path, dataset, prefix), mode='r') as fd:
        for line in fd:
            if not line: continue
            h, r, t = line.strip().split('\t')
            tuples.add('{}||{}||{}'.format(h, r, t).replace(' ', '_'))
            data.append((h, r, t, 1.0))

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
            data.append((h, r + '_ext', t, score))

fw = open('{}/{}/scores.txt'.format(data_path, dataset), mode='r')
for h, r, t, score in data:
    fw.write('{}\t{}\t{}\t{}\n'.format(h, r, t, score))
fw.close()
