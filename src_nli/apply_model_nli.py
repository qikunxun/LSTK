import logging
import json
import torch
import os
import sys
from transformers import AutoTokenizer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_checkpoint = 'bert-base-uncased'
use_chinese = False
if use_chinese: model_checkpoint = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
batch_size = 8
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

def load_data(filename, relation2id):
    logging.info("Loading data")
    out = []
    with open(filename, mode='r') as fd:
        for line in tqdm(fd.readlines()):
            if not line: continue
            data = json.loads(line.strip())
            h = data['h']
            t = data['t']
            texts = data['texts']
            if len(texts) == 0: continue
            out.append({'h': h, 't': t, 'texts': texts})
            # if len(out) > 100: break
    return out

def batch_iter(data, relation2id, batch_size=4):
    global tokenizer
    for i, item in enumerate(data):
        batch = []
        h = item['h']
        t = item['t']
        texts = item['texts']
        for j, relation in enumerate(relation2id):
            batch.append(relation)
            if len(batch) == batch_size or j == (len(relation2id) - 1):
                input_ids = []
                token_type_ids = []
                attention_mask = []
                indices = []
                index = 0
                for r in batch:
                    triple = '{}\t{}\t{}'.format(h, r, t)
                    for text in texts:
                        encoding = tokenizer(triple.replace('_', ' ').replace('\t', ' '), text, padding='max_length', truncation=True, max_length=128)
                        input_ids.append(encoding['input_ids'])
                        token_type_ids.append(encoding['token_type_ids'])
                        attention_mask.append(encoding['attention_mask'])
                    indices.append((index, index + len(texts)))
                    index += len(texts)
                yield {'input_ids': torch.tensor(input_ids), 'token_type_ids': torch.tensor(token_type_ids),
                       'attention_mask': torch.tensor(attention_mask)}, indices
                batch = []
dataset = sys.argv[1]
relation2id, id2relation = get_relations('../datasets/{}/relations.txt'.format(dataset))
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1)
device = torch.device("cuda")
model.load_state_dict(torch.load('../models/model_{}.pt'.format(dataset), map_location=torch.device('cpu')))
model.to(device)

model.eval()

for prefix in ['train', 'valid', 'test']:
    evaluate_data = load_data('../datasets/{}/{}_triples.txt'.format(dataset, prefix), relation2id)
    progress_bar = tqdm(range(len(evaluate_data) * len(relation2id) // batch_size))
    data_iter = batch_iter(evaluate_data, relation2id, batch_size=batch_size)
    y_pred = []
    for batch, indices in data_iter:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits.squeeze(dim=-1)
        logits_items = []
        for start, end in indices:
            logits_item = logits[start: end]
            logits_items.append(torch.max(logits_item, dim=0)[0])
        logits_items = torch.stack(logits_items, dim=0)
        logits_items = torch.sigmoid(logits_items)
        y_pred.append(logits_items.detach().cpu())
        progress_bar.update(1)
    y_pred = torch.cat(y_pred, dim=0)
    y_score = y_pred
    y_pred = (y_pred > 0.5).int()
    fw = open('../datasets/{}/{}_triple_scores.txt'.format(dataset, prefix), mode='w')
    for i, items in enumerate(evaluate_data):
        for j, r in enumerate(relation2id):
            h = items['h']
            t = items['t']
            triple = '{}||{}||{}'.format(h, r, t)
            fw.write('{}\t{}\t{}\n'.format(triple, y_score[i * len(relation2id) + j], y_pred[i * len(relation2id) + j]))
    fw.close()











