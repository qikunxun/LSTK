import logging
import json
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from transformers import AutoTokenizer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_checkpoint = 'bert-base-uncased'
use_chinese = False
if use_chinese: model_checkpoint = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
batch_size = 32
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

def load_data(filename):
    logging.info("Loading data")
    out = []
    with open(filename, mode='r') as fd:
        for line in tqdm(fd.readlines()):
            if not line: continue
            data = json.loads(line.strip())
            triple = data['triple']
            texts = data['texts']
            if len(texts) == 0: continue
            out.append({'triple': triple, 'texts': texts})
    return out

def get_facts(data):
    facts = set()
    for item in data:
        facts.add(item['triple'])
    return facts


def batch_iter(data, id2relation, facts, batch_size=4):
    global tokenizer
    np.random.shuffle(data)
    batch = []
    for i, item in enumerate(data):
        triple = item['triple']
        h, r, t = triple.split('\t')
        texts = item['texts']
        for j in range(len(id2relation)):
            encodings = []
            r_new = id2relation[j]
            label = 0
            if r.replace('_', ' ') == r_new: label = 1
            triple_new = '{}\t{}\t{}'.format(h, r_new, t)
            for text in texts:
                encoding = tokenizer(triple_new.replace('_', ' ').replace('\t', ' '), text, padding='max_length',
                                     truncation=True, max_length=128)
                encodings.append(encoding)
            batch.append({'encodings': encodings, 'label': label})
            if len(batch) == batch_size or (i == (len(data) - 1) and j == len(id2relation) - 1):
                input_ids = []
                token_type_ids = []
                attention_mask = []
                label_ids = []
                indices = []
                index = 0
                for item_batch in batch:
                    encodings = item_batch['encodings']
                    for encoding in encodings:
                        input_ids.append(encoding['input_ids'])
                        token_type_ids.append(encoding['token_type_ids'])
                        attention_mask.append(encoding['attention_mask'])
                    label_ids.append(item_batch['label'])
                    indices.append((index, index + len(encodings)))
                    index += len(encodings)
                yield {'input_ids': torch.tensor(input_ids), 'token_type_ids': torch.tensor(token_type_ids),
                       'attention_mask': torch.tensor(attention_mask)}, torch.tensor(label_ids), indices
                batch = []
dataset = sys.argv[1]
relation2id, id2relation = get_relations('../datasets/{}/relations.txt'.format(dataset))
train_data = load_data('../datasets/{}/train_nli.txt'.format(dataset))
facts = get_facts(train_data)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_data) * len(relation2id)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=int(num_training_steps * 0.1), num_training_steps=num_training_steps
)
device = torch.device("cuda")
model.to(device)

model.train()
progress_bar = tqdm(range(num_training_steps // batch_size))
for epoch in range(num_epochs):
    data_iter = batch_iter(train_data, id2relation, facts, batch_size=batch_size)
    for batch, labels, indices in data_iter:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = labels.float().to(device)
        outputs = model(**batch)
        logits = outputs.logits.squeeze(dim=-1)
        logits_items = []
        for start, end in indices:
            logits_item = logits[start: end]
            logits_items.append(torch.max(logits_item, dim=0)[0])
        logits_items = torch.stack(logits_items, dim=0)
        loss = F.binary_cross_entropy_with_logits(logits_items, labels)
        print(loss.item())
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
torch.save(model.state_dict(), '../models/model_{}.pt'.format(dataset))