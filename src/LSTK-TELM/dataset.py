import os
import numpy as np
from tqdm import tqdm


class Triple:
    def __init__(self, h, r, t):
        self.h = h
        self.r = r
        self.t = t

    def get_triple(self):
        return self.h, self.r, self.t


class Dataset:

    def __init__(self, kg_path, batch_size, target_relation, negative_sampling=True, option=None):
        self.option = option
        self.kg_path = kg_path
        self.batch_size = batch_size
        self.target_relation = target_relation
        self.kg, self.id2entity, self.entity2id, self.id2relation, self.relation2id, \
        self.train_kg_ori, self.train_kg_inv = self.load_kg_all(kg_path)
        self.graph, self.graph_entity, self.relation_tail, self.relation_head = self.build_graph()
        self.targets_h = self.get_targets_head()
        self.targets_t = self.get_targets_tail()
        self.negative_sampling = negative_sampling
        self.soft_relations = self.get_soft_relations(kg_path)

    def load_kg_all(self, kg_path):
        kg = []
        entities = set()
        relations = set()
        fact_path = os.path.join(kg_path, 'facts.txt')
        train_path = os.path.join(kg_path, 'train.txt')
        # entity_path = os.path.join(kg_path, 'entities.dict')
        relation_path = os.path.join(kg_path, 'relations.dict')

        if os.path.exists(fact_path):
            with open(fact_path, mode='r') as fd:
                for line in fd:
                    if not line: continue
                    items = line.strip().split('\t')
                    if len(items) != 3: continue
                    h, r, t = items
                    kg.append(Triple(h, r, t))
                    kg.append(Triple(t, 'INV' + r, h))
                    entities.add(h)
                    entities.add(t)
                    relations.add(r)
                    relations.add('INV' + r)
        train_triples_ori = []
        train_triples_inv = []
        with open(train_path, mode='r') as fd:
            for line in fd:
                if not line: continue
                items = line.strip().split('\t')
                if len(items) != 3: continue
                h, r, t = items
                kg.append(Triple(h, r, t))
                kg.append(Triple(t, 'INV' + r, h))
                entities.add(h)
                entities.add(t)
                relations.add(r)
                relations.add('INV' + r)
                if r == self.target_relation:
                    train_triples_inv.append(Triple(t, 'INV' + r, h))
                    train_triples_ori.append(Triple(h, r, t))

        id2entity, entity2id = {}, {}
        for entity in entities:
            entity2id[entity] = len(entity2id)
            id2entity[entity2id[entity]] = entity

        id2relation, relation2id = {}, {}
        with open(relation_path, mode='r') as fd:
            for line in fd:
                if not line: continue
                idx, relation = line.strip().split('\t')
                relation2id[relation] = int(idx)
                id2relation[int(idx)] = relation
        length = len(relation2id)
        for i in range(length):
            relation2id['INV' + id2relation[i]] = len(relation2id)
            id2relation[len(relation2id) - 1] = 'INV' + id2relation[i]
        return kg, id2entity, entity2id, id2relation, relation2id, train_triples_ori, train_triples_inv

    def batch_iter(self):
        for i in range(self.option.iteration_per_batch):
            if len(self.train_kg_ori) == 0: break
            batch_ori = np.random.choice(self.train_kg_ori, self.batch_size, replace=True)
            entity_tail = {}
            triples = []
            for triple_batch in batch_ori:
                h, r, t = triple_batch.get_triple()
                h_hat, t_hat = None, None
                if self.negative_sampling:
                    t_hat = self.select_random_entity_tail(self.targets_t, t, h)
                    h_hat = self.select_random_entity_head(self.targets_h, h, t)
                    if t_hat not in entity_tail: entity_tail[t_hat] = len(entity_tail)
                if t not in entity_tail: entity_tail[t] = len(entity_tail)
                triples.append((h, r, t, h_hat, t_hat))
            yield triples, entity_tail, False

            entity_tail = {}
            triples = []
            batch_inv = np.random.choice(self.train_kg_inv, self.batch_size, replace=True)
            for triple_batch in batch_inv:
                h, r, t = triple_batch.get_triple()
                h_hat, t_hat = None, None
                if self.negative_sampling:
                    t_hat = self.select_random_entity_tail(self.targets_t, t, h)
                    h_hat = self.select_random_entity_head(self.targets_h, h, t)
                    if t_hat not in entity_tail: entity_tail[t_hat] = len(entity_tail)
                if t not in entity_tail: entity_tail[t] = len(entity_tail)
                triples.append((h, r, t, h_hat, t_hat))
            yield triples, entity_tail, True

    def build_graph(self):
        graph = {}
        graph_entity = {}
        relation_head = {}
        relation_tail = {}
        for triple in self.kg:
            h, r, t = triple.get_triple()
            if h not in graph:
                graph[h] = {r: [t]}
                graph_entity[h] = {t: [r]}
            else:
                if r in graph[h]:
                    graph[h][r].append(t)
                else:
                    graph[h][r] = [t]

                if t in graph_entity[h]:
                    graph_entity[h][t].append(r)
                else:
                    graph_entity[h][t] = [r]
            if r == self.target_relation and t not in relation_tail: relation_tail[t] = len(relation_tail)
            if r == self.target_relation and h not in relation_head: relation_head[h] = len(relation_head)
        return graph, graph_entity, relation_tail, relation_head

    def get_targets_tail(self):
        targets = []
        relation_tail = sorted(self.relation_tail.items(), key=lambda d: d[1])
        for item in relation_tail:
            targets.append(item[0])
        return targets

    def get_targets_head(self):
        targets = []
        entities = sorted(self.entity2id.items(), key=lambda d: d[1])
        for item in entities:
            targets.append(item[0])
        return targets

    def select_random_entity_tail(self, targets, cur, h, max_deep=5):
        selected_entity = None
        count = 0
        while selected_entity is None or selected_entity == cur or \
                (h in self.graph and self.target_relation in self.graph[h]
                 and selected_entity in self.graph[h][self.target_relation]):
            if count == max_deep: break
            selected_entity = np.random.choice(targets, 1)[0]
            count += 1
        return selected_entity

    def select_random_entity_head(self, targets, cur, t, max_deep=5):
        selected_entity = None
        count = 0
        while selected_entity is None or selected_entity == cur or \
                (selected_entity in self.graph and self.target_relation in self.graph[selected_entity]
                 and t in self.graph[selected_entity][self.target_relation]):
            # if count == max_deep: break
            selected_entity = np.random.choice(targets, 1)[0]
            count += 1
        return selected_entity

    def get_soft_relations(self, kg_path):
        fact_path = os.path.join(kg_path, 'train_triple_scores.txt')
        soft_relations = {}
        with open(fact_path, mode='r') as fd:
            for i, line in enumerate(tqdm(fd.readlines())):
                if not line: continue
                items = line.strip().split('\t')
                if len(items) != 3: continue
                triple, score, pred = items
                score = float(score)
                if score < self.option.delta: continue
                triple = triple.replace(' ', '_')
                h, r, t = triple.split('||')
                if h not in self.entity2id or t not in self.entity2id: continue
                soft_relations[triple] = score
        scores = sorted(soft_relations.items(), key=lambda x: x[1], reverse=True)
        soft_relations = {}
        for triple, score in scores[:]:
            h, r, t = triple.split('||')
            triple_inv = '{}||{}||{}'.format(t, 'INV' + r, h)
            soft_relations[triple] = score
            soft_relations[triple_inv] = score

        print('Size of soft relation:', len(soft_relations))
        return soft_relations

    def get_soft_relations_test(self):
        fact_path = os.path.join(self.kg_path, 'test_triple_scores.txt')
        soft_relations = {}
        with open(fact_path, mode='r') as fd:
            for i, line in enumerate(tqdm(fd.readlines())):
                if not line: continue
                items = line.strip().split('\t')
                if len(items) != 3: continue
                triple, score, pred = items
                score = float(score)
                if score < self.option.delta: continue
                triple = triple.replace(' ', '_')
                soft_relations[triple] = score
        scores = sorted(soft_relations.items(), key=lambda x: x[1], reverse=True)
        soft_relations = self.soft_relations
        for triple, score in scores[:]:
            h, r, t = triple.split('||')
            triple_inv = '{}||{}||{}'.format(t, 'INV' + r, h)
            soft_relations[triple] = score
            soft_relations[triple_inv] = score

        print('Size of soft relation:', len(soft_relations))
        return soft_relations
