import pickle as pkl
import json
import random

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.cluster import KMeans

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        vec = pkl.load(f)
        return vec


def dump_pickle(file_path, obj):
    with open(file_path, 'wb') as f:
        pkl.dump(obj, f)


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def dump_json(file_path, obj):
    with open(file_path, 'w') as f:
        json.dump(obj, f)


def remove_unseen_relation(data, seen_relations, dataset='fewrel'):
    cleaned_data = []
    for data in data:
        neg_cands = [cand for cand in data[1] if cand in seen_relations]
        if len(neg_cands) > 0:
            if dataset == 'fewrel':
                cleaned_data.append([data[0], neg_cands, data[2], data[3], data[4], data[5]])
            else:
                cleaned_data.append([data[0], neg_cands, data[2]])
        else:
            if dataset == 'fewrel':
                cleaned_data.append([data[0], data[1][-2:], data[2], data[3], data[4], data[5]])
            else:
                cleaned_data.append([data[0], data[1][-2:], data[2]])
    return cleaned_data

def ranking_sequence(sequence):
    word_lengths = torch.tensor([len(sentence) for sentence in sequence])
    rankedi_word, indexs = word_lengths.sort(descending = True)
    ranked_indexs, inverse_indexs = indexs.sort()
    #print(indexs)
    sequence = [sequence[i] for i in indexs]
    return sequence, inverse_indexs

def get_que_embed(model, sample_list, all_relations, batch_size, device,
                  before_alignment=False):
    ret_que_embeds = []
    for i in range((len(sample_list)-1)//batch_size+1):
        samples = sample_list[i*batch_size:(i+1)*batch_size]
        questions = []
        for item in samples:
            this_question = torch.tensor(item[2], dtype=torch.long).to(device)
            questions.append(this_question)
        #print(len(questions))
        model.init_hidden(device, len(questions))
        ranked_questions, alignment_question_indexs = \
            ranking_sequence(questions)
        question_lengths = [len(question) for question in ranked_questions]
        #print(ranked_questions)
        pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)
        que_embeds = model.compute_que_embed(pad_questions, question_lengths,
                                             alignment_question_indexs, None, before_alignment)
        ret_que_embeds.append(que_embeds.detach().cpu().numpy())
    return np.concatenate(ret_que_embeds)

# get the embedding of relations. If before_alignment is False, then the
# embedding after the alignment model will be returned. Otherwise, the embedding
# before the alignment model will be returned
def get_rel_embed(model, sample_list, all_relations, alignment_model, batch_size, device,
                  before_alignment=False):
    ret_rel_embeds = []
    for i in range((len(sample_list)-1)//batch_size+1):
        samples = sample_list[i*batch_size:(i+1)*batch_size]
        relations = []
        for item in samples:
            this_relation = torch.tensor(all_relations[item[0]],
                                         dtype=torch.long).to(device)
            relations.append(this_relation)
        #print(len(relations))
        model.init_hidden(device, len(relations))
        ranked_relations, alignment_relation_indexs = \
            ranking_sequence(relations)
        relation_lengths = [len(relation) for relation in ranked_relations]
        #print(ranked_relations)
        pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
        rel_embeds = model.compute_rel_embed(pad_relations, relation_lengths,
                                             alignment_relation_indexs,
                                             alignment_model, before_alignment)
        ret_rel_embeds.append(rel_embeds.detach().cpu().numpy())
    return np.concatenate(ret_rel_embeds)

def select_data(model, samples, num_sel_data, all_relations, batch_size, device):
    que_embeds = get_que_embed(model, samples, all_relations, batch_size, device)  # sentence embedding，400d
    que_embeds = preprocessing.normalize(que_embeds)  # sklearn normalize
    #print(que_embeds[:5])
    num_clusters = min(num_sel_data, len(samples))  # cluster samples into min(num_sel_data, len(samples))clusters, get one for each cluster as memory
    distances = KMeans(n_clusters=num_clusters,
                       random_state=0).fit_transform(que_embeds)
    selected_samples = []
    for i in range(num_clusters):
        sel_index = np.argmin(distances[:,i])
        selected_samples.append(samples[sel_index])
    return selected_samples

def random_select_data(current_train_data, task_memory_size):
    return random.sample(current_train_data, task_memory_size)

# process the data by adding questions
def process_testing_samples(sample_list, all_relations, device):
    questions = []
    relations = []
    gold_relation_indexs = []
    relation_set_lengths = []
    for sample in sample_list:
        question = torch.tensor(sample[2], dtype=torch.long).to(device)
        #print(relations[sample[0]])
        #print(sample)
        gold_relation_indexs.append(sample[0])
        neg_relations = [torch.tensor(all_relations[index - 1],
                                      dtype=torch.long).to(device)
                         for index in sample[1]]
        relation_set_lengths.append(len(neg_relations))
        relations += neg_relations
        #questions += [question for i in range(relation_set_lengths[-1])]
        questions += [question] * relation_set_lengths[-1]
    return gold_relation_indexs, questions, relations, relation_set_lengths

def process_samples(sample_list, all_relations, device):
    questions = []
    relations = []
    relation_set_lengths = []
    for sample in sample_list:
        question = torch.tensor(sample[2], dtype=torch.long).to(device)
        #print(relations[sample[0]])
        #print(sample)
        pos_relation = torch.tensor(all_relations[sample[0] - 1],
                                    dtype=torch.long).to(device)  # pos tensor
        neg_relations = [torch.tensor(all_relations[index - 1],
                                      dtype=torch.long).to(device)
                         for index in sample[1]]  # candidate neg tensor
        relation_set_lengths.append(len(neg_relations)+1)
        relations += [pos_relation] + neg_relations  # merge
        #questions += [question for i in range(relation_set_lengths[-1])]
        questions += [question] * relation_set_lengths[-1]
    return questions, relations, relation_set_lengths

def ranking_sequence(sequence):
    word_lengths = torch.tensor([len(sentence) for sentence in sequence])
    ranked_word, indexs = word_lengths.sort(descending = True)
    ranked_indexs, inverse_indexs = indexs.sort()
    #print(indexs)
    sequence = [sequence[i] for i in indexs]
    return sequence, inverse_indexs

def append_log(file_name, line):
    with open(file_name, 'a+') as f:
        f.writelines(line + '\n')
        f.flush()