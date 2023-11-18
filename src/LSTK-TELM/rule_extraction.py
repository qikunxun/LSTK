import os
import json
import sys
import torch
import pickle as pkl
from model import ICLMModel
from tqdm import tqdm

beam_size = int(sys.argv[5])

class Option(object):
    def __init__(self, path):
        with open(os.path.join(path, 'option.txt'), mode='r') as f:
            self.__dict__ = json.load(f)

def reverse(x2id):
    id2x = {}
    for x in x2id:
        id2x[x2id[x]] = x
    return id2x

def load_kg_form_pkl(file_path, target_relation):
    with open(file_path + 'kg_{}.pkl'.format(target_relation.replace('/', '|')), mode='rb') as fd:
        kg = pkl.load(fd)
    with open(file_path + 'entity2id_{}.pkl'.format(target_relation.replace('/', '|')), mode='rb') as fd:
        entity2id = pkl.load(fd)
    with open(file_path + 'relation2id_{}.pkl'.format(target_relation.replace('/', '|')), mode='rb') as fd:
        relation2id = pkl.load(fd)

    return kg, entity2id, relation2id



def get_beam(indices, t, beam_size, r, T, all_indices):
    beams = indices[:, :beam_size]
    for l in range(indices.shape[0]):
        index = indices[l]
        j = 0
        for i in range(index.shape[-1]):
            beams[l][j] = index[i]
            j += 1
            if j == beam_size: break

    return beams

def get_states(indices, scores):
    states = torch.zeros(indices.shape)
    for l in range(indices.shape[0]):
        states[l] = torch.index_select(scores[l], -1, indices[l])
    return states

def transform_score(x, T):
    one = torch.autograd.Variable(torch.Tensor([1]))
    zero = torch.autograd.Variable(torch.Tensor([0]).detach())
    return torch.minimum(torch.maximum(x / T, zero), one)

def analysis(id2relation, relation2id, flag, option, model_save_path):
    thr = 0.5
    T = 4
    model = ICLMModel(len(relation2id), option.step, option.length, T, 0.2, False)

    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    model.eval()
    n = len(relation2id)
    w = torch.softmax(model.w[-1][:, :n], dim=-1)
    w_ = torch.softmax(model.w[-1][:, n:], dim=-1)
    alpha_ori = torch.sigmoid(model.alpha[-1][flag])
    beta_ori = torch.sigmoid(model.beta[-1][flag])
    alpha = (alpha_ori >= thr).float()
    beta = (beta_ori >= thr).float()
    atom_type = alpha * beta
    scores = torch.zeros(option.length, 2 * n + 1)
    for l in range(option.length):
        if alpha[l] == 0 and beta[l] == 0:
            scores[l, -1] = 1 - model.activation(alpha_ori[l] + beta_ori[l])
        elif alpha[l] == 1 and beta[l] == 0:
            scores[l][:n] = w[l]
        elif alpha[l] == 0 and beta[l] == 1:
            scores[l][n:-1] = w_[l]
        else:
            tmp = torch.stack([w[l], w_[l]], dim=-1)
            scores[l][:n] = torch.max(tmp, dim=-1)[0]
    scores = torch.log(scores)
    indices_order = torch.argsort(scores, dim=-1, descending=True)
    all_indices = []
    indices = get_beam(indices_order, option.step - 2, beam_size, len(relation2id), option.step, all_indices)
    states = get_states(indices, scores)
    all_indices.append(indices)
    for t in range(option.step - 3, -1, -1):
        w = torch.softmax(model.w[t][:, :n], dim=-1)
        w_ = torch.softmax(model.w[t][:, n:], dim=-1)
        alpha_ori = torch.sigmoid(model.alpha[t][flag])
        beta_ori = torch.sigmoid(model.beta[t][flag])
        alpha = (alpha_ori >= thr).float()
        beta = (beta_ori >= thr).float()
        atom_type = alpha * beta
        scores = torch.zeros(option.length, 2 * n + 1)
        for l in range(option.length):
            if alpha[l] == 0 and beta[l] == 0:
                scores[l][-1] = 1 - model.activation(alpha_ori[l] + beta_ori[l])
            elif alpha[l] == 1 and beta[l] == 0:
                scores[l][:n] = w[l]
            elif alpha[l] == 0 and beta[l] == 1:
                scores[l][n:-1] = w_[l]
            else:
                tmp = torch.stack([w[l], w_[l]], dim=-1)
                scores[l][:n] = torch.max(tmp, dim=-1)[0]
        scores = torch.log(scores)
        scores = states.unsqueeze(dim=-1) + scores.unsqueeze(dim=1)
        scores = scores.view(option.length, -1)
        indices_order = torch.argsort(scores, dim=-1, descending=True)
        topk_indices = get_beam(indices_order, t, beam_size, len(relation2id), option.step, all_indices)
        states = get_states(topk_indices, scores)
        all_indices.append(topk_indices)
    outputs = torch.zeros(option.length, option.step - 1, beam_size).long()
    p = torch.zeros(option.length, beam_size).long()
    for beam in range(beam_size):
        p[:, beam] = beam
    for t in range(option.step - 1):
        for l in range(option.length):
            for beam in range(beam_size):
                c = int(all_indices[option.step - t - 2][l][p[l][beam]] % (2 * n + 1))
                outputs[l][t][beam] = c
                p_new = int(all_indices[option.step - t - 2][l][p[l][beam]] / (2 * n + 1))
                p[l][beam] = p_new
    all_rules = []
    for l in range(option.length):
        rule = '{}(x, y)<-'.format(option.target_relation)
        rules = [rule] * beam_size
        counts = torch.zeros(option.length, beam_size)
        for beam in range(beam_size):
            y = ''
            for t in range(option.step - 2, -1, -1):
                alpha = (torch.sigmoid(model.alpha[t][flag]) >= thr).float()
                beta = (torch.sigmoid(model.beta[t][flag]) >= thr).float()
                atom_type = alpha * beta
                c = int(outputs[l][t][beam])
                if not (alpha[l] == 0 and beta[l] == 0):
                    if atom_type[l] == 0:
                        tmp = id2relation[c]
                    else:
                        if c >= n:
                            tmp_str = id2relation[c - n]
                            tmp_text = id2relation[c]
                        else:
                            tmp_str = id2relation[c]
                            tmp_text = id2relation[c + n]
                        tmp = '({} ∨ {})'.format(tmp_str, tmp_text)
                    x = 'x'
                    if counts[l][beam] > 0: x = 'z_{}'.format(int(counts[l][beam]) - 1)
                    y = 'z_{}'.format(int(counts[l][beam]))
                    if t == 0: y = 'y'
                    if t > 0:
                        flag_tmp = True
                        for j in range(t):
                            if outputs[l][j][beam] != n * n: flag_tmp = False
                            break
                        if flag_tmp: y = 'y'
                    output = tmp + '({}, {})'.format(x, y)
                    counts[l][beam] += 1
                else:
                    identity = 'x'
                    if t != option.step - 2 and y != '': identity = y
                    output = 'Identity({}, {})'.format(identity, identity)
                end = ''
                if t > 0: end = ' ∧ '
                rules[beam] += output + end
        all_rules.append(rules)

    ids_sort = torch.argsort(model.weight.squeeze(dim=-1), descending=True)
    flag_ = 'ori'
    if flag: flag_ = 'inv'
    fw = open('./{}/rules-{}-{}.txt'.format(option.exps_dir, option.target_relation.replace('/', '|'), flag_), mode='w')
    for i, ids in enumerate(ids_sort):
        data = {'rank': (i + 1), 'id': int(ids), 'rules': all_rules[int(ids)], 'weight': float(torch.tanh(model.weight[int(ids)]))}
        fw.write(json.dumps(data, ensure_ascii=False) + '\n')
        print('Rank: {}, id: {}, Weight: {}, Rule: {}'.format((i + 1), ids, float(torch.tanh(model.weight[int(ids)])), all_rules[int(ids)]))
    fw.close()



if __name__ == '__main__':
    exps_dir = sys.argv[1]
    exp_name = sys.argv[2]
    target_relation = sys.argv[3].replace('/', '|')
    flag = sys.argv[4]
    option = Option('{}/{}-{}-{}/'.format(exps_dir, exp_name, target_relation, 'ori'))
    train_kg, entity2id, relation2id = load_kg_form_pkl('{}/'.format(option.exp_dir), option.target_relation)
    id2relation = reverse(relation2id)
    for i in range(len(relation2id)):
        id2relation[len(relation2id) + i] = id2relation[i] + '_text'
    flag_id = 0
    if flag == 'inv': flag_id = 1
    analysis(id2relation, relation2id, flag_id, option,
             '{}/model_{}.pt'.format(option.exp_dir, option.target_relation.replace('/', '|')))