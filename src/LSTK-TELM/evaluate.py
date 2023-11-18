import os
import json
import sys
import torch
import pickle as pkl
from model import TELMModel
from tqdm import tqdm
from dataset import Triple, Dataset

batch_size = int(sys.argv[4])
inverse = int(sys.argv[5])
if inverse == 1:
    inverse = True
else:
    inverse = False
print(inverse)
use_gpu = True
if use_gpu: os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[6]


class Option(object):
    def __init__(self, path):
        with open(os.path.join(path, 'option.txt'), mode='r') as f:
            self.__dict__ = json.load(f)


def load_kg_form_pkl(file_path, target_relation):
    with open(file_path + 'kg_{}.pkl'.format(target_relation.replace(' ', '_').replace('/', '|')), mode='rb') as fd:
        kg = pkl.load(fd)
    with open(file_path + 'entity2id_{}.pkl'.format(target_relation.replace(' ', '_').replace('/', '|')),
              mode='rb') as fd:
        entity2id = pkl.load(fd)
    with open(file_path + 'relation2id_{}.pkl'.format(target_relation.replace(' ', '_').replace('/', '|')),
              mode='rb') as fd:
        relation2id = pkl.load(fd)
    return kg, entity2id, relation2id


def save_tail(tail2id, flag, file_path=None):
    if file_path is None: return
    with open(os.path.join(file_path, 'tail2id_{}.pkl'.format(flag)), mode='wb') as fw:
        pkl.dump(tail2id, fw)


def load_data(data_path):
    data_ori = []
    data_inv = []
    with open(data_path, mode='r') as fd:
        for line in fd:
            if not line: continue
            items = line.strip().split('\t')
            if len(items) != 3: continue
            h, r, t = items
            data_ori.append(Triple(h, r, t))
            data_inv.append(Triple(t, 'INV' + r, h))
    return data_ori, data_inv


def extend_graph(graph_entity, valid_data, test_data):
    data = valid_data.copy()
    data.extend(test_data)
    for triple in data:
        h, r, t = triple.get_triple()
        if h not in graph_entity:
            graph_entity[h] = {t: [r]}
        else:
            if t in graph_entity[h]:
                graph_entity[h][t].append(r)
            else:
                graph_entity[h][t] = [r]


def reverse(x2id):
    id2x = {}
    for x in x2id:
        id2x[x2id[x]] = x
    return id2x


def build_graph(kg):
    graph = {}
    graph_entity = {}
    for triple in kg:
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
    return graph, graph_entity


def get_head(heads, kg):
    entity2id_head = {}
    id2entity_head = {}
    for h in heads:
        entity2id_head[h] = len(entity2id_head)
        id2entity_head[entity2id_head[h]] = h
    for h, r, t in kg:
        if h not in entity2id_head: continue
        if t in entity2id_head: continue
        entity2id_head[t] = len(entity2id_head)
        id2entity_head[entity2id_head[t]] = t
    return entity2id_head, id2entity_head


def get_init_matrix(kg, soft_relations, entity2id, relation2id, triples):
    i_x = []
    i_y = []
    v = []
    cur_set = set()
    cur_ent_set = set()
    for x, r, y, x_hat, y_hat in triples:
        if r.startswith('INV'):
            cur_set.add('{}||{}||{}'.format(x, r, y))
            cur_set.add('{}||{}||{}'.format(y, r[3:], x))
        else:
            cur_set.add('{}||{}||{}'.format(x, r, y))
            cur_set.add('{}||{}||{}'.format(y, 'INV' + r, x))
        cur_ent_set.add(x)
        cur_ent_set.add(y)

    records = set()
    for triple in kg:
        h, r, t = triple.get_triple()
        flag = '{}||{}||{}'.format(h, r, t)
        if flag in cur_set: continue
        if h not in entity2id or t not in entity2id: continue
        entity_a = entity2id[h]
        entity_b = entity2id[t]
        relation = relation2id[r]
        record = '{}\t{}'.format(entity_a * (2 * len(relation2id) + 1) + relation, entity_b)
        if record not in records:
            i_x.append(entity_a * (2 * len(relation2id) + 1) + relation)
            i_y.append(entity_b)
            v.append(1)
            records.add(record)

        record = '{}\t{}'.format(entity_a * (2 * len(relation2id) + 1) + 2 * len(relation2id), entity_a)
        if record not in records:
            i_x.append(entity_a * (2 * len(relation2id) + 1) + 2 * len(relation2id))
            i_y.append(entity_a)
            v.append(1)
            records.add(record)
    for triple in soft_relations:
        h, r, t = triple.split('||')
        # if triple in cur_set: continue
        score = soft_relations[triple]
        if h not in entity2id or t not in entity2id: continue
        # if h in cur_ent_set and t in cur_ent_set: continue
        entity_a = entity2id[h]
        entity_b = entity2id[t]
        relation = relation2id[r] + len(relation2id)
        record = '{}\t{}'.format(entity_a * (2 * len(relation2id) + 1) + relation, entity_b)
        if record not in records:
            i_x.append(entity_a * (2 * len(relation2id) + 1) + relation)
            i_y.append(entity_b)
            v.append(score)
            records.add(record)
    return torch.LongTensor([i_x, i_y]), torch.FloatTensor(v)


def init_matrix(matrix, kg, soft_relations, entity2id, relation2id, relation_tail):
    # print('Processing Matirx(shape={})'.format(matrix.shape))
    for triple in kg:
        h, r, t = triple.get_triple()
        if h not in entity2id: continue
        if t not in relation_tail: continue
        entity_a = entity2id[h]
        entity_b = relation_tail[t]
        relation = relation2id[r]
        matrix[entity_a][entity_b][relation] = 1
        matrix[entity2id[t]][entity_b][len(relation2id)] = 1

    for triple in soft_relations:
        h, r, t = triple.split('||')
        if h not in entity2id: continue
        if t not in relation_tail: continue
        score = soft_relations[triple]
        entity_a = entity2id[h]
        entity_b = relation_tail[t]
        relation = relation2id[r] + len(relation2id)
        matrix[entity_a][entity_b][relation] = score
        matrix[entity2id[t]][entity_b][len(relation2id)] = 1


def init_mask(mask, triples, entity2id, relation2id, relation_tail):
    # print('Processing Mask(shape={})'.format(mask.shape))
    for h, r, t, x_hat, y_hat in triples:
        if h not in entity2id or t not in entity2id: continue
        if t not in relation_tail: continue
        entity_a = entity2id[h]
        entity_b = relation_tail[t]
        relation = relation2id[r]
        mask[entity_a][entity_b][relation] = 1
        mask[entity_a][entity_b][relation + len(relation2id)] = 1
        if h in relation_tail:
            if r.startswith('INV'):
                r_ = r[3:]
            else:
                r_ = 'INV' + r
            relation_inv = relation2id[r_]
            entity_a_inv = entity2id[t]
            entity_b_inv = relation_tail[h]
            mask[entity_a_inv][entity_b_inv][relation_inv] = 1
            mask[entity_a_inv][entity_b_inv][relation_inv + len(relation2id)] = 1


def evaluate(id2entity, entity2id, id2relation, relation2id, soft_relations, train_kg, valid_data, test_data, option,
             target_relation, model_save_path, raw=False):
    print('Entity Num:', len(entity2id))
    print('Relation Num:', len(relation2id))
    print('Train KG Size:', len(train_kg))
    print('Eval KG Size:', len(test_data[0]))
    print('Target relation:', target_relation)
    graph, graph_entity = build_graph(train_kg)
    extend_graph(graph_entity, valid_data[0], test_data[0])
    extend_graph(graph_entity, valid_data[1], test_data[1])
    model = TELMModel(len(relation2id), option.step, option.length,
                      option.tau_1, option.tau_2, use_gpu)

    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    # for parameter in model.parameters():
    #     print(parameter)
    if use_gpu: model = model.cuda()
    model.eval()
    kg = train_kg.copy()
    kg.extend(valid_data[0])
    kg.extend(valid_data[1])
    kg.extend(test_data[0])
    kg.extend(test_data[1])
    entity2id_new = entity2id.copy()
    for triple in valid_data[0]:
        h, r, t = triple.get_triple()
        if h not in entity2id_new:
            entity2id_new[h] = len(entity2id_new)
        if t not in entity2id_new:
            entity2id_new[t] = len(entity2id_new)
    for triple in test_data[0]:
        h, r, t = triple.get_triple()
        if h not in entity2id_new:
            entity2id_new[h] = len(entity2id_new)
        if t not in entity2id_new:
            entity2id_new[t] = len(entity2id_new)
    mrr_tail = 0
    mrr_head = 0
    hit_1_head = 0
    hit_1_tail = 0
    hit_3_head = 0
    hit_3_tail = 0
    hit_10_head = 0
    hit_10_tail = 0
    count_head = 0
    count_tail = 0
    for triple in tqdm(test_data[0]):
        h, r, t = triple.get_triple()
        if r != target_relation: continue
        if h not in entity2id_new or t not in entity2id_new: continue
        i, v = get_init_matrix(kg, soft_relations, entity2id_new, relation2id, [(h, r, t, None, None)])
        matrix_all = torch.sparse.FloatTensor(i, v,
                                              torch.Size([len(entity2id_new) * (2 * len(relation2id) + 1),
                                                          len(entity2id_new)]))
        matrix = torch.zeros([len(entity2id_new), 1, (2 * len(relation2id) + 1)])
        entity2id_tail = {t: 0}
        init_matrix(matrix, kg, soft_relations, entity2id_new, relation2id, entity2id_tail)
        # print('sum', matrix.sum())
        mask = torch.zeros([len(entity2id_new), len(entity2id_tail), 2 * len(relation2id) + 1])
        init_mask(mask, [(h, r, t, None, None)], entity2id_new, relation2id, entity2id_tail)
        matrix = matrix * (1 - mask)
        matrix = matrix.view(-1, (2 * len(relation2id) + 1)).to_sparse()
        if option.use_gpu:
            matrix = matrix.cuda()
            matrix_all = matrix_all.cuda()
        all_states = []
        for step in range(option.step):
            state = model(matrix, matrix_all, all_states, step, entity2id_new, False)
            print(state.sum())
            all_states.append(state)
        total_states = all_states[-1].cpu().detach()
        truth_score_ori = total_states[entity2id_new[h]][0]
        index_head = torch.argsort(total_states[:, 0], descending=True).cpu().numpy()
        if raw:
            rank = torch.sum((total_states[:, 0] >= truth_score_ori).int())
        else:
            scores_head = total_states[:, 0].clone()
            for tail in graph_entity[t]:
                if tail not in entity2id_new: continue
                r_tmp = r
                if r_tmp.startswith('INV'):
                    r_tmp = r_tmp[3:]
                else:
                    r_tmp = 'INV' + r_tmp
                if r_tmp in graph_entity[t][tail]: scores_head[entity2id_new[tail]] = -1e20
            n = torch.sum((scores_head == truth_score_ori).int()) + 1
            m = torch.sum((scores_head > truth_score_ori).int())
            rank = m + (n + 1) / 2

        mrr_head += 1 / rank
        if rank <= 1:
            hit_1_head += 1

        if rank <= 3:
            hit_3_head += 1

        if rank <= 10:
            hit_10_head += 1
        print((h, r, t), truth_score_ori, rank)
        count_head += 1
    if count_head > 0:
        mrr_head /= count_head
        hit_1_head /= count_head
        hit_3_head /= count_head
        hit_10_head /= count_head

    print('Target Relation: {}'.format(target_relation))
    print('# of evaluated triples: {}'.format(count_head))
    print('Mrr_head: {}'.format(mrr_head))
    print('Hit@1_head: {}'.format(hit_1_head))
    print('Hit@3_head: {}'.format(hit_3_head))
    print('Hit@10_head: {}'.format(hit_10_head))
    for triple in tqdm(test_data[1]):
        h, r, t = triple.get_triple()
        if r != 'INV' + target_relation: continue
        if h not in entity2id_new or t not in entity2id_new: continue
        i, v = get_init_matrix(kg, soft_relations, entity2id_new, relation2id, [(h, r, t, None, None)])
        matrix_all = torch.sparse.FloatTensor(i, v,
                                              torch.Size([len(entity2id_new) * (2 * len(relation2id) + 1),
                                                          len(entity2id_new)]))
        matrix = torch.zeros([len(entity2id_new), 1, (2 * len(relation2id) + 1)])
        entity2id_tail = {t: 0}
        init_matrix(matrix, kg, soft_relations, entity2id_new, relation2id, entity2id_tail)
        # print('sum', matrix.sum())
        mask = torch.zeros([len(entity2id_new), len(entity2id_tail), 2 * len(relation2id) + 1])
        init_mask(mask, [(h, r, t, None, None)], entity2id_new, relation2id, entity2id_tail)
        matrix = matrix * (1 - mask)
        matrix = matrix.view(-1, (2 * len(relation2id) + 1)).to_sparse()
        if option.use_gpu:
            matrix = matrix.cuda()
            matrix_all = matrix_all.cuda()
        all_states = []
        for step in range(option.step):
            state = model(matrix, matrix_all, all_states, step, entity2id_new, True)
            # print(state.sum())
            all_states.append(state)
        total_states = all_states[-1].cpu().detach()
        truth_score_inv = total_states[entity2id_new[h]][0]
        index_head = torch.argsort(total_states[:, 0], descending=True).cpu().numpy()
        if raw:
            rank = torch.sum((total_states[:, 0] >= truth_score_inv).int())
        else:
            scores_tail = total_states[:, 0].clone()
            for tail in graph_entity[t]:
                if tail not in entity2id_new: continue
                r_tmp = r
                if r_tmp.startswith('INV'):
                    r_tmp = r_tmp[3:]
                else:
                    r_tmp = 'INV' + r_tmp
                if r_tmp in graph_entity[t][tail]: scores_tail[entity2id_new[tail]] = -1e20
            n = torch.sum((scores_tail == truth_score_inv).int()) + 1
            m = torch.sum((scores_tail > truth_score_inv).int())
            rank = m + (n + 1) / 2

        mrr_tail += 1 / rank
        if rank <= 1:
            hit_1_tail += 1

        if rank <= 3:
            hit_3_tail += 1

        if rank <= 10:
            hit_10_tail += 1

        count_tail += 1
    if count_tail > 0:
        mrr_tail /= count_tail
        hit_1_tail /= count_tail
        hit_3_tail /= count_tail
        hit_10_tail /= count_tail
    print('Target Relation: {}'.format(target_relation))
    print('# of evaluated triples: {}'.format(count_head))
    print('Mrr_head: {}'.format(mrr_head))
    print('Mrr_tail: {}'.format(mrr_tail))
    print('Hit@1_head: {}'.format(hit_1_head))
    print('Hit@1_tail: {}'.format(hit_1_tail))
    print('Hit@3_head: {}'.format(hit_3_head))
    print('Hit@3_tail: {}'.format(hit_3_tail))
    print('Hit@10_head: {}'.format(hit_10_head))
    print('Hit@10_tail: {}'.format(hit_10_tail))
    print('Mrr: {}'.format((mrr_head + mrr_tail) / 2))
    print('Hit@1: {}'.format((hit_1_head + hit_1_tail) / 2))
    print('Hit@3: {}'.format((hit_3_head + hit_3_tail) / 2))
    print('Hit@10: {}'.format((hit_10_head + hit_10_tail) / 2))
    print('=' * 50)
    return mrr_tail, mrr_head, hit_1_head, hit_1_tail, hit_3_head, hit_3_tail, hit_10_head, hit_10_tail, count_head, count_tail


if __name__ == '__main__':
    data_path = sys.argv[1]
    valid_data = load_data(os.path.join(data_path, 'valid.txt'))
    test_data = load_data(os.path.join(data_path, 'test.txt'))
    total_mrr_tail, total_mrr_head, total_hit_1_head, total_hit_1_tail, total_hit_3_head, total_hit_3_tail, \
    total_hit_10_head, total_hit_10_tail = 0, 0, 0, 0, 0, 0, 0, 0
    total_size = len(test_data[0])
    with open(os.path.join(data_path, 'relations.dict'), mode='r') as fd:
        for line in fd:
            if not line: continue
            items = line.strip().split('\t')
            idx, relation = items
            option = Option(
                '{}/{}-{}-ori'.format(sys.argv[2], sys.argv[3], relation.replace(' ', '_').replace('/', '|')))
            dataset = Dataset(option.data_dir, option.batch_size, option.target_relation, option.negative_sampling,
                              option)
            soft_relations = dataset.get_soft_relations_test()
            train_kg, entity2id, relation2id = load_kg_form_pkl('{}/'.format(option.exp_dir),
                                                                option.target_relation.replace(' ', '_').replace('/',
                                                                                                                 '|'))
            id2entity = reverse(entity2id)
            id2relation = reverse(relation2id)
            mrr_tail, mrr_head, hit_1_head, hit_1_tail, hit_3_head, hit_3_tail, \
            hit_10_head, hit_10_tail, count_head, count_tail = evaluate(id2entity, entity2id, id2relation,
                                                                        relation2id, soft_relations, train_kg,
                                                                        valid_data, test_data, option, relation,
                                                                        '{}/model_{}.pt'.format(option.exp_dir,
                                                                                                option.target_relation.replace(
                                                                                                    ' ', '_').replace(
                                                                                                    '/', '|'),
                                                                                                option.target_relation.replace(
                                                                                                    ' ', '_').replace(
                                                                                                    '/', '|')))
            total_mrr_tail += mrr_tail * count_tail / total_size
            total_mrr_head += mrr_head * count_head / total_size
            total_hit_1_head += hit_1_head * count_head / total_size
            total_hit_1_tail += hit_1_tail * count_tail / total_size
            total_hit_3_head += hit_3_head * count_head / total_size
            total_hit_3_tail += hit_3_tail * count_tail / total_size
            total_hit_10_head += hit_10_head * count_head / total_size
            total_hit_10_tail += hit_10_tail * count_tail / total_size
    print('Target Relation: ALL')
    print('# of evaluated triples: {}'.format(total_size))
    print('Mrr_head: {}'.format(total_mrr_head))
    print('Mrr_tail: {}'.format(total_mrr_tail))
    print('Hit@1_head: {}'.format(total_hit_1_head))
    print('Hit@1_tail: {}'.format(total_hit_1_tail))
    print('Hit@3_head: {}'.format(total_hit_3_head))
    print('Hit@3_tail: {}'.format(total_hit_3_tail))
    print('Hit@10_head: {}'.format(total_hit_10_head))
    print('Hit@10_tail: {}'.format(total_hit_10_tail))
    print('Mrr: {}'.format((total_mrr_head + total_mrr_tail) / 2))
    print('Hit@1: {}'.format((total_hit_1_head + total_hit_1_tail) / 2))
    print('Hit@3: {}'.format((total_hit_3_head + total_hit_3_tail) / 2))
    print('Hit@10: {}'.format((total_hit_10_head + total_hit_10_tail) / 2))
    print('=' * 50)
