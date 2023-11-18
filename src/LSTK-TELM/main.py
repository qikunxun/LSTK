import argparse
import time
import random
import json
import pickle as pkl
import torch
import os
import numpy as np
from model import TELMModel
from tqdm import tqdm
from dataset import Dataset, Triple


class Option(object):
    def __init__(self, d):
        self.__dict__ = d

    def save(self):
        if not os.path.exists(self.exps_dir):
            os.mkdir(self.exps_dir)
        flag = '-ori'
        if self.inverse:
            flag = '-inv'
        self.exp_dir = os.path.join(self.exps_dir, self.exp_name + '-' +
                                    self.target_relation.replace(' ', '_').replace('/', '|') + flag)
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        with open(os.path.join(self.exp_dir, "option.txt"), "w") as f:
            json.dump(self.__dict__, f, indent=1)
        return True


def set_seed(option):
    random.seed(option.seed)
    np.random.seed(option.seed)
    torch.manual_seed(option.seed)
    os.environ['PYTHONHASHSEED'] = str(option.seed)
    if option.use_gpu: torch.cuda.manual_seed_all(option.seed)


def save_data(target_relation, kg, entity2id, relation2id, file_path=None):
    print(len(kg))
    with open(os.path.join(file_path, 'kg_{}.pkl'.format(target_relation.replace(' ', '_').replace('/', '|'))),
              mode='wb') as fw:
        pkl.dump(kg, fw)
    with open(os.path.join(file_path, 'entity2id_{}.pkl'.format(target_relation.replace(' ', '_').replace('/', '|'))),
              mode='wb') as fw:
        pkl.dump(entity2id, fw)
    with open(os.path.join(file_path, 'relation2id_{}.pkl'.format(target_relation.replace(' ', '_').replace('/', '|'))),
              mode='wb') as fw:
        pkl.dump(relation2id, fw)


def build_graph(kg, target_relation):
    graph = {}
    graph_entity = {}
    relation_tail = {}
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
        if r == target_relation and t not in relation_tail: relation_tail[t] = len(relation_tail)
    return graph, graph_entity, relation_tail


def get_init_matrix(kg, soft_relations, entity2id, relation2id, triples):
    i_x = []
    i_y = []
    v = []
    cur_set = set()
    for x, r, y, x_hat, y_hat in triples:
        if r.startswith('INV'):
            cur_set.add('{}||{}||{}'.format(x, r, y))
            cur_set.add('{}||{}||{}'.format(y, r[3:], x))
        else:
            cur_set.add('{}||{}||{}'.format(x, r, y))
            cur_set.add('{}||{}||{}'.format(y, 'INV' + r, x))

    records = set()
    for triple in kg:
        h, r, t = triple.get_triple()
        flag = '{}||{}||{}'.format(h, r, t)
        if flag in cur_set: continue
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
        if t not in relation_tail: continue
        entity_a = entity2id[h]
        entity_b = relation_tail[t]
        relation = relation2id[r]
        matrix[entity_a][entity_b][relation] = 1
        matrix[entity2id[t]][entity_b][len(relation2id)] = 1

    for triple in soft_relations:
        h, r, t = triple.split('||')
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


def norm(probs):
    sum_all = np.sum(probs)
    probs_new = np.ones_like(probs) / probs.shape[0]
    if sum_all > 0:
        probs_new = probs / sum_all
    return probs_new


def select_random_entity_by_probs(targets, probs):
    selected_entity = np.random.choice(targets, 1, p=probs)
    return selected_entity[0]


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def load_data(data_path, target_relation):
    data_ori = []
    data_inv = []
    with open(data_path, mode='r') as fd:
        for line in fd:
            if not line: continue
            items = line.strip().split('\t')
            if len(items) != 3: continue
            h, r, t = items
            if r != target_relation: continue
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


def valid_process(valid_data, valid_data_inv, model, train_kg, flag, soft_relations, entity2id,
                  relation2id, graph_entity, option, raw=False, batch_size=1):
    if len(valid_data) == 0: return 0, 0, 0, 0
    model.eval()
    kg = train_kg.copy()
    kg.extend(valid_data)
    kg.extend(valid_data_inv)
    entity2id_new = entity2id.copy()
    for triple in valid_data:
        h, r, t = triple.get_triple()
        if h not in entity2id_new:
            entity2id_new[h] = len(entity2id_new)
        if t not in entity2id_new:
            entity2id_new[t] = len(entity2id_new)
    mrr = 0
    hit_1 = 0
    hit_3 = 0
    hit_10 = 0
    count = 0
    batch = []
    for k, triple in enumerate(tqdm(valid_data)):
        h, r, t = triple.get_triple()
        if h not in entity2id_new or t not in entity2id_new: continue
        batch.append((h, r, t, None, None))
        if len(batch) == batch_size or k == len(valid_data) - 1:
            entity2id_tail = {}
            for h, r, t, _, _ in batch:
                if t in entity2id_tail: continue
                entity2id_tail[t] = len(entity2id_tail)
            i, v = get_init_matrix(kg, soft_relations, entity2id_new, relation2id, batch)
            matrix_all = torch.sparse.FloatTensor(i, v,
                                                  torch.Size([len(entity2id_new) * (2 * len(relation2id) + 1),
                                                              len(entity2id_new)]))
            matrix = torch.zeros([len(entity2id_new), len(entity2id_tail), (2 * len(relation2id) + 1)])
            init_matrix(matrix, kg, soft_relations, entity2id_new, relation2id, entity2id_tail)
            mask = torch.zeros([len(entity2id_new), len(entity2id_tail), 2 * len(dataset.relation2id) + 1])
            init_mask(mask, batch, entity2id_new, dataset.relation2id, entity2id_tail)
            matrix = matrix * (1 - mask)
            matrix = matrix.view(-1, (2 * len(relation2id) + 1)).to_sparse()
            if option.use_gpu:
                matrix = matrix.cuda()
                matrix_all = matrix_all.cuda()
            all_states = []
            for step in range(option.step):
                state = model(matrix, matrix_all, all_states, step, entity2id_new, flag, is_training=False)
                all_states.append(state)
            total_states = all_states[-1].cpu().detach()
            for h, r, t, _, _ in batch:
                truth_score_ori = total_states[entity2id_new[h]][entity2id_tail[t]]
                index_head = torch.argsort(total_states[:, entity2id_tail[t]], descending=True).cpu().numpy()
                if raw:
                    rank = torch.sum((total_states[:, entity2id_tail[t]] >= truth_score_ori).int())
                else:
                    scores_head = total_states[:, entity2id_tail[t]].clone()
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

                mrr += 1 / rank
                if rank <= 1:
                    hit_1 += 1

                if rank <= 3:
                    hit_3 += 1

                if rank <= 10:
                    hit_10 += 1
                count += 1
            batch = []
    if count > 0:
        mrr /= count
        hit_1 /= count
        hit_3 /= count
        hit_10 /= count
    print('Valid Count:{}\tMrr:{}\tHit@1:{}\tHit@3:{}\tHit@10:{}'.format(count, mrr, hit_1, hit_3, hit_10))
    return mrr, hit_1, hit_3, hit_10


def main(dataset, valid_data, test_data, option):
    print('Current Time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print('Modeling target relation: {}'.format(option.target_relation))
    print('Entity Num:', len(dataset.entity2id))
    print('Relation Num:', len(dataset.relation2id))
    print('Train KG Size:', len(dataset.train_kg_ori))
    graph_entity = dataset.graph_entity
    extend_graph(graph_entity, valid_data[0], test_data[0])
    extend_graph(graph_entity, valid_data[1], test_data[1])
    model = TELMModel(len(dataset.relation2id), option.step, option.length,
                      option.tau_1, option.tau_2, option.use_gpu)
    if option.use_gpu: model = model.cuda()
    for parameter in model.parameters():
        print(parameter)
    optimizer = torch.optim.Adam(model.parameters(), lr=option.learning_rate)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=option.learning_rate * 0.8)
    end_flag = False
    max_score = 0
    max_record = {'mrr': 0, 'hit_1': 0, 'hit_3': 0, 'hit_10': 0, 'epoch': 0}
    never_saved = True
    for e in range(option.max_epoch):
        model.train()
        total_loss = 0
        if end_flag: break
        for k, batch in enumerate(dataset.batch_iter()):
            triples, entity_tail, flag = batch
            i, v = get_init_matrix(dataset.kg, dataset.soft_relations, dataset.entity2id, dataset.relation2id, triples)
            matrix_all = torch.sparse.FloatTensor(i, v,
                                                  torch.Size(
                                                      [len(dataset.entity2id) * (2 * len(dataset.relation2id) + 1),
                                                       len(dataset.entity2id)]))
            matrix = torch.zeros([len(dataset.entity2id), len(entity_tail), 2 * len(dataset.relation2id) + 1])
            mask = torch.zeros([len(dataset.entity2id), len(entity_tail), 2 * len(dataset.relation2id) + 1])
            init_matrix(matrix, dataset.kg, dataset.soft_relations, dataset.entity2id, dataset.relation2id, entity_tail)
            init_mask(mask, triples, dataset.entity2id, dataset.relation2id, entity_tail)
            matrix = matrix * (1 - mask)
            matrix = matrix.view(-1, 2 * len(dataset.relation2id) + 1).to_sparse()
            if option.use_gpu:
                matrix = matrix.cuda()
                matrix_all = matrix_all.cuda()
            all_states = []
            for t in range(option.step):
                state = model(matrix, matrix_all, all_states, t, dataset.entity2id, flag, is_training=True)
                if t < option.step - 1: print(state.sum())
                all_states.append(state)
            loss = 0
            for x, r, y, x_hat, y_hat in tqdm(triples):
                if option.negative_sampling:
                    loss += model.negative_loss(
                        torch.unsqueeze(all_states[option.step - 1][dataset.entity2id[x]][entity_tail[y]], dim=0),
                        torch.unsqueeze(all_states[option.step - 1][dataset.entity2id[x_hat]][entity_tail[y]], dim=0))
                else:
                    logit_mask = torch.zeros([len(dataset.entity2id)])
                    if r.startswith('INV'):
                        inv_r = r[3:]
                    else:
                        inv_r = 'INV' + r
                    for ent in dataset.graph[y][inv_r]:
                        if ent == x: continue
                        logit_mask[dataset.entity2id[ent]] = 1
                    loss += model.log_loss(torch.unsqueeze(all_states[option.step - 1][:, entity_tail[y]], dim=0),
                                           dataset.entity2id[x], logit_mask)

            print('Epoch: {}, Batch: {}, Loss: {}'.format(e, k, loss.item()))
            total_loss += loss.item()
            if loss > option.threshold:
                loss.backward()
                # torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=10)
                optimizer.step()
                sch.step(loss)
                optimizer.zero_grad()
        print('Epoch: {}, Total Loss: {}'.format(e, total_loss))
        if option.early_stop and (e + 1) % 5 == 0:
            mrr_ori, hit_1_ori, hit_3_ori, hit_10_ori = valid_process(valid_data[0], valid_data[1], model, dataset.kg,
                                                                      False,
                                                                      dataset.soft_relations, dataset.entity2id,
                                                                      dataset.relation2id,
                                                                      graph_entity, option)
            mrr_inv, hit_1_inv, hit_3_inv, hit_10_inv = valid_process(valid_data[1], valid_data[0], model, dataset.kg,
                                                                      True,
                                                                      dataset.soft_relations, dataset.entity2id,
                                                                      dataset.relation2id,
                                                                      graph_entity, option)
            mrr = (mrr_ori + mrr_inv) / 2
            hit_1 = (hit_1_ori + hit_1_inv) / 2
            hit_3 = (hit_3_ori + hit_3_inv) / 2
            hit_10 = (hit_10_ori + hit_10_inv) / 2
            if mrr > max_score:
                max_score = mrr
                max_record['mrr'] = mrr
                max_record['hit_1'] = hit_1
                max_record['hit_3'] = hit_3
                max_record['hit_10'] = hit_10
                max_record['epoch'] = e
                torch.save(model.state_dict(),
                           os.path.join(option.exp_dir, 'model_{}.pt'.format(
                               option.target_relation.replace(' ', '_').replace('/', '|'))))
                never_saved = False
            print('=' * 100)
            print('Valid Max Score : Epoch:{}\tMrr:{}\tHit@1:{}\tHit@3:{}\tHit@10:{}'.format(max_record['epoch'],
                                                                                             max_record['mrr'],
                                                                                             max_record['hit_1'],
                                                                                             max_record['hit_3'],
                                                                                             max_record['hit_10']))
        if never_saved:
            torch.save(model.state_dict(),
                       os.path.join(option.exp_dir,
                                    'model_{}.pt'.format(option.target_relation.replace(' ', '_').replace('/', '|'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment setup")
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--exps_dir', default=None, type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--use_gpu', default=False, action="store_true")
    parser.add_argument('--gpu_id', default=4, type=int)
    # model architecture
    parser.add_argument('--length', default=20, type=int)
    parser.add_argument('--step', default=4, type=int)  # step=T+1
    parser.add_argument('--tau_1', default=2, type=float)
    parser.add_argument('--tau_2', default=0.2, type=float)
    parser.add_argument('--delta', default=0.5, type=float)
    parser.add_argument('--target_relation', default=None, type=str)
    parser.add_argument('--with_constrain', default=True, action="store_true")
    parser.add_argument('--inverse', default=False, action="store_true")
    # optimization
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--iteration_per_batch', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--early_stop', default=False, action="store_true")
    parser.add_argument('--threshold', default=1e-6, type=float)
    parser.add_argument('--negative_sampling', default=False, action="store_true")
    parser.add_argument('--seed', default=1234, type=int)

    d = vars(parser.parse_args())
    option = Option(d)
    option.tag = time.strftime("%y-%m-%d %H:%M:%S")
    bl = option.save()
    print("Option saved.")

    if option.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(option.gpu_id)
    set_seed(option)
    dataset = Dataset(option.data_dir, option.batch_size, option.target_relation, option.negative_sampling, option)
    save_data(option.target_relation, dataset.kg, dataset.entity2id, dataset.relation2id, option.exp_dir)
    valid_data = load_data(os.path.join(option.data_dir, 'valid.txt'), option.target_relation)
    test_data = load_data(os.path.join(option.data_dir, 'test.txt'), option.target_relation)
    main(dataset, valid_data, test_data, option)
