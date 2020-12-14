# this script is used to create new dataset with entities and positions
import json
from data_loader import remove_invalid_token, remove_return_sym, lower_case
from tqdm import tqdm

fewrel_train_file = './dataset/fewrel/train_wiki.json'
fewrel_valid_file = './dataset/fewrel/val_wiki.json'
fewrel_relation_file = './dataset/fewrel/all_relations.json'

with open(fewrel_train_file, 'r') as f:
    fewrel_train_data = json.load(f)

with open(fewrel_valid_file, 'r') as f:
    fewrel_valid_data = json.load(f)

with open(fewrel_relation_file, 'r') as f:
    fewrel_relations = json.load(f)

id2rel = {}
rel2id = {}
for wiki_rel_id in fewrel_relations:
    idx = int(fewrel_relations[wiki_rel_id]['index']) + 1
    id2rel[idx] = wiki_rel_id
    rel2id[wiki_rel_id] = idx

lifelong_fewrel_train_file = './dataset/training_data.txt'
lifelong_fewrel_valid_file = './dataset/val_data.txt'

lifelong_fewrel_train_file_with_entity = './dataset/training_data_with_entity.txt'
lifelong_fewrel_valid_file_with_entity = './dataset/test_data_with_entity.txt'

lifelong_fewrel_train_data = []
lifelong_fewrel_valid_data = []
# lifelong_fewrel_train_data_with_entity = []
# lifelong_fewrel_valid_data_with_entity = []


with open(lifelong_fewrel_train_file, 'r') as f:
    for line in f:
        lifelong_fewrel_train_data.append(line)
print('Load %d training instance' % len(lifelong_fewrel_train_data))
with open(lifelong_fewrel_valid_file, 'r') as f:
    for line in f:
        lifelong_fewrel_valid_data.append(line)
print('Load %d validation instance' % len(lifelong_fewrel_valid_data))


def process_lifelong_data(fewrel_data):
    data_with_entity = []
    for i in tqdm(range(len(fewrel_data))):
        line = fewrel_data[i]
        items = line.split('\t')
        rel_idx = int(items[0])  # int
        candidate_rel_idx = [int(idx) for idx in items[1].split()]
        tokens = remove_invalid_token(remove_return_sym(items[2]).split())

        if rel_idx in id2rel:
            wiki_rel_id = id2rel[rel_idx]
        else:
            print('relation %s not found' % rel_idx)

        if wiki_rel_id in fewrel_train_data:
            instance_pool = fewrel_train_data[wiki_rel_id]
        elif wiki_rel_id in fewrel_valid_data:
            instance_pool = fewrel_valid_data[wiki_rel_id]
        else:
            print('wiki relation %s not found' % wiki_rel_id)

        current_instance = None
        for instance_item in instance_pool:
            _tokens = lower_case(remove_invalid_token(instance_item['tokens']))
            if ' '.join(_tokens) == ' '.join(tokens):
                current_instance = instance_item
                break

        if current_instance is None:
            print('instance: [%s] not found' % line)

        h = current_instance['h'][0]
        h_pos = current_instance['h'][-1]
        t = current_instance['t'][0]
        t_pos = current_instance['t'][-1]

        item_with_entity = [rel_idx, candidate_rel_idx, tokens, h, h_pos, t, t_pos]

        data_with_entity.append(item_with_entity)
    return data_with_entity

print('processing training data')
lifelong_fewrel_train_data_with_entity = process_lifelong_data(lifelong_fewrel_train_data)
print('processing validation data')
lifelong_fewrel_valid_data_with_entity = process_lifelong_data(lifelong_fewrel_valid_data)

def dump_new_data(data_with_entity, file_path):
    with open(file_path, 'w') as f:
        for i in tqdm(range(len(data_with_entity))):
            item = data_with_entity[i]
            f.writelines('%d\t%s\t%s\t%s\t%s\t%s\t%s\n' % (
                item[0],
                ' '.join([str(cand_rel) for cand_rel in item[1]]),
                ' '.join(item[2]),
                item[3],
                ' '.join([str(pos) for pos in item[4][0]]),
                item[5],
                ' '.join([str(pos) for pos in item[6][0]]),
            ))

print('dump training data with entity')
dump_new_data(lifelong_fewrel_train_data_with_entity, lifelong_fewrel_train_file_with_entity)
print('dump valid data with entity')
dump_new_data(lifelong_fewrel_valid_data_with_entity, lifelong_fewrel_valid_file_with_entity)