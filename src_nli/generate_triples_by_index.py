import json
import sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import pymetis
from nltk.stem import LancasterStemmer

lancaster_stemmer = LancasterStemmer()

def get_word_stem(words):
    global lancaster_stemmer
    word_stems = []
    for word in words:
        word_stems.append(lancaster_stemmer.stem(word))
    return word_stems

def search_by_jaccard(r, texts, topk=10):
    filtered = []
    relation_words = set(get_word_stem(r.split(' ')))
    for text in texts:
        text_words = set(get_word_stem(text.split(' ')))
        sim = len(relation_words & text_words) / len(relation_words | text_words)
        filtered.append((text, sim))
    sorted(filtered, key=lambda x: x[1])
    results = []
    for t, s in filtered[:topk]:
        results.append(t)
    return results

def search_from_texts(query, indics, index_value):
    flag = True
    for word in query:
        if word not in indics:
            flag = False
            break
        if index_value is None:
            index_value = indics[word]
        else:
            index_value = list_intersection(indics[word], index_value)
        if len(index_value) == 0:
            flag = False
            break
    if not flag: return None
    return index_value

def search_by_metis(texts):
    if len(texts) <= 10:
        return texts
    adj, xadj, w = [], [], []
    end_node = 0
    xadj.append(0)
    for i, text_a in enumerate(texts):
        text_a = set(get_word_stem(text_a.split(' ')))
        for j, text_b in enumerate(texts):
            text_b = set(get_word_stem(text_b.split(' ')))
            score = len(text_a & text_b) / len(text_a | text_b)
            if score == 0: continue
            adj.append(j)
            w.append(int(score * 100))
            end_node += 1
        xadj.append(end_node)
    xadj = np.array(xadj)
    adj = np.array(adj)
    w = np.array(w)
    # print(len(texts), xadj.shape, adj.shape, w.shape)
    nparts = int(len(texts) / 10)
    (edgecuts, parts) = pymetis.part_graph(nparts=nparts, xadj=xadj, adjncy=adj, eweights=w)
    labels = {}
    for i, part in enumerate(parts):
        if part not in labels: labels[part] = []
        text = texts[i]
        labels[part].append(text)
    scores = {}
    max_score = -1
    for label in labels:
        text_tmp = labels[label]
        score = 0
        count = 0
        if len(text_tmp) > 1:
            for text_a in text_tmp:
                text_a = set(get_word_stem(text_a.split(' ')))
                for text_b in text_tmp:
                    text_b = set(get_word_stem(text_b.split(' ')))
                    if text_a == text_b: continue
                    score += len(text_a & text_b) / len(text_a | text_b)
                    count += 1
            score = score / (count + 1e-6)
        if score > max_score: max_score = score
        scores[score] = label
    max_label = scores[max_score]
    return labels[max_label]

def get_relations(filename):
    relation2id = {}
    id2relation = {}
    with open(filename, mode='r') as fd:
        for line in fd:
            if not line: continue
            relation = line.strip()
            relation = relation.replace('_', ' ')
            relation2id[relation] = len(relation2id)
            id2relation[relation2id[relation]] = relation
    return relation2id, id2relation

def build_index(corpus):
    indices = {}
    for i, sentence in enumerate(tqdm(corpus)):
        words = get_word_stem(sentence.split(' '))
        # words = sentence.split(' ')
        for word in words:
            if word in indices:
                indices[word].add(i)
            else:
                indices[word] = {i}
    return indices

def list_intersection(a, b):
    return list(set(a).intersection(set(b)))

def loop_entities(h, h_index, entities, indices, sentences):
    results = []
    for t in entities:
        if h == t: continue
        t_words = get_word_stem(t.split(' '))
        t_index = search_from_texts(t_words, indices, h_index)
        if t_index is None: continue
        texts = []
        for index in t_index:
            text = ' ' + sentences[index] + ' '
            if ' {} '.format(h) not in text or ' {} '.format(t) not in text: continue
            texts.append(sentences[index])
        if len(texts) == 0: continue
        if len(texts) > 20:
            if len(texts) > 1000:
                texts = np.random.choice(texts, 1000, replace=False)
            texts = search_by_metis(texts)
        results.append((t, texts))
    return results

if __name__ == '__main__':
    dataset = sys.argv[1]
    corpus_train = []
    with open('../datasets/{}/corpus_train.txt'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            corpus_train.append(line.strip())

    print('Training corpus size:', len(corpus_train))
    corpus_valid = []
    with open('../datasets/{}/corpus_valid.txt'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            corpus_valid.append(line.strip())

    print('Valid corpus size:', len(corpus_valid))
    corpus_test = []
    with open('../datasets/{}/corpus_test.txt'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            corpus_test.append(line.strip())

    print('Testing corpus size:', len(corpus_test))

    corpus = corpus_train
    indices = build_index(corpus)
    entities = set()
    with open('../datasets/{}/train.txt'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            items = line.strip().split('\t')
            h, r, t = items
            h = h.replace('_', ' ')
            t = t.replace('_', ' ')
            entities.add(h)
            entities.add(t)
            # if len(entities) > 100: break
    relation2id, id2relation = get_relations('../datasets/{}/relations.txt'.format(dataset))
    results = []
    batch_size = 32
    pool = Pool(processes=8)
    for i, h in enumerate(tqdm(entities)):
        h_words = get_word_stem(h.split(' '))
        h_index = search_from_texts(h_words, indices, None)
        if h_index is None: continue
        result = pool.apply_async(loop_entities, (h, h_index, entities, indices, corpus))
        results.append((h, result))
    pool.close()
    pool.join()
    fw = open('../datasets/{}/train_triples.txt'.format(dataset), mode='w')
    for h, result in results:
        items = result.get()
        for t, texts in items:
            if len(texts) == 0: continue
            fw.write(json.dumps({'h': h, 't': t, 'texts': texts}) + '\n')
    fw.close()

    corpus = corpus_train + corpus_valid
    indices = build_index(corpus)
    entities_valid = set()
    with open('../datasets/{}/valid.txt'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            items = line.strip().split('\t')
            h, r, t = items
            h = h.replace('_', ' ')
            t = t.replace('_', ' ')
            entities_valid.add(h)
            entities_valid.add(t)

    results = []
    batch_size = 32
    pool = Pool(processes=8)
    entities_all = entities | entities_valid
    for i, h in enumerate(tqdm(entities_valid)):
        h_words = get_word_stem(h.split(' '))
        h_index = search_from_texts(h_words, indices, None)
        if h_index is None: continue
        result = pool.apply_async(loop_entities, (h, h_index, entities_all, indices, corpus))
        results.append((h, result))
    pool.close()
    pool.join()
    fw = open('../datasets/{}/valid_triples.txt'.format(dataset), mode='w')
    for h, result in results:
        items = result.get()
        for t, texts in items:
            if len(texts) == 0: continue
            fw.write(json.dumps({'h': h, 't': t, 'texts': texts}) + '\n')
    fw.close()

    corpus = corpus_train + corpus_valid + corpus_test
    indices = build_index(corpus)
    entities_test = set()
    with open('../datasets/{}/test.txt'.format(dataset), mode='r') as fd:
        for line in fd:
            if not line: continue
            items = line.strip().split('\t')
            h, r, t = items
            h = h.replace('_', ' ')
            t = t.replace('_', ' ')
            entities_test.add(h)
            entities_test.add(t)

    results = []
    batch_size = 32
    pool = Pool(processes=8)
    entities_all = entities | entities_valid | entities_test
    for i, h in enumerate(tqdm(entities_test)):
        h_words = get_word_stem(h.split(' '))
        h_index = search_from_texts(h_words, indices, None)
        if h_index is None: continue
        result = pool.apply_async(loop_entities, (h, h_index, entities_all, indices, corpus))
        results.append((h, result))
    pool.close()
    pool.join()
    fw = open('../datasets/{}/test_triples.txt'.format(dataset), mode='w')
    for h, result in results:
        items = result.get()
        for t, texts in items:
            if len(texts) == 0: continue
            fw.write(json.dumps({'h': h, 't': t, 'texts': texts}) + '\n')
    fw.close()