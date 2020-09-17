import os
import pickle
import random

from utils import *
import numpy as np
import wordninja
import re

def lower_case(token_list):
    return [str.lower(token) for token in token_list]

def remove_return_sym(string):
    return string.rstrip('\n')

def remove_invalid_token(token_list):
    invalid_chars = ['\xa0', '\n', ' ', '\u3000', '\u2005']
    for invalid_char in invalid_chars:
        token_list = [char for char in token_list if invalid_char not in char]
    return token_list

def read_data(file_path):
    if not os.path.exists(file_path):
        raise Exception('No such file %s' % file_path)
    tmp_dir = os.path.join(os.path.dirname(file_path), 'tmp')

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    tmp_file_path = os.path.join(tmp_dir, '%s.pkl' % file_path.split('/')[-1].split('.')[0])

    if os.path.exists(tmp_file_path):
        return read_pickle(tmp_file_path)
    else:
        with open(file_path, 'r', encoding='utf8') as f:
            data_dict = {}
            data_list = []
            for line in f:
                items = remove_return_sym(line).split('\t')
                rel_idx = int(items[0])
                if items[1] != 'noNegativeAnswer':
                    candidate_rel_idx = [int(idx) for idx in items[1].split()]
                    tokens = remove_invalid_token(remove_return_sym(items[2]).split())
                    if len(items) > 3:
                        h = items[3]  # entity string not need to separate cause we will not directly use it
                        h_pos = [int(idx) for idx in items[4].split()]
                        t = items[5]
                        t_pos = [int(idx) for idx in items[6].split()]

                        data_item = [rel_idx, candidate_rel_idx, tokens, h, h_pos, t, t_pos]
                        data_list.append(data_item)
                        if rel_idx not in data_dict:
                            data_dict[rel_idx] = [data_item]
                        else:
                            data_dict[rel_idx].append(data_item)
                    else:
                        data_item = [rel_idx, candidate_rel_idx, tokens]
                        data_list.append(data_item)
                        if rel_idx not in data_dict:
                            data_dict[rel_idx] = [data_item]
                        else:
                            data_dict[rel_idx].append(data_item)

            dump_pickle(tmp_file_path, (data_list, data_dict))
            return data_list, data_dict

def read_relation(file_path):
    if not os.path.exists(file_path):
        raise Exception('No such file %s' % file_path)

    with open(file_path, 'r', encoding='utf8') as f:
        relation_list = []
        index = 1
        relation_dict = {}
        for line in f:
            relation_name = remove_return_sym(line)
            relation_list.append(relation_name)
            relation_dict[relation_name] = index
            index += 1

        return relation_list, relation_dict

def read_glove(glove_file):
    if not os.path.exists(glove_file):
        raise Exception('No such file %s' % glove_file)
    tmp_dir = os.path.join(os.path.dirname(glove_file), 'tmp')

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    tmp_file_path = os.path.join(tmp_dir, '%s.pkl' % glove_file.split('/')[-1].split('.')[0])

    if os.path.exists(tmp_file_path):
        glove_vocabulary, glove_embedding = read_pickle(tmp_file_path)
        return glove_vocabulary, glove_embedding

    glove_vocabulary = []
    glove_embedding = {}
    with open(glove_file, 'r', encoding='utf8') as file_in:
        for line in file_in:
            items = line.split()
            word = items[0]
            glove_vocabulary.append(word)
            glove_embedding[word] = np.asarray(items[1:], dtype='float32')
    dump_pickle(tmp_file_path, (glove_vocabulary, glove_embedding))

    # relation extraction
    return glove_vocabulary, glove_embedding

def concat_words(words):
    if len(words) > 0:
        return_str = words[0]
        for word in words[1:]:
            return_str += '_' + word
        return return_str
    else:
        return ''

def split_relation_into_words(relation, glove_vocabulary):
    word_list = []
    relation_list = []
    # some relation will have fours parts, where the first part looks like
    # "base". We only choose the last three parts
    for word_seq in relation.split("/")[-3:]:
        new_word_list = []
        #for word in word_seq.split("_"):
        for word in re.findall(r"[\w']+", word_seq):
            #print(word)
            if word not in glove_vocabulary:
                new_word_list += wordninja.split(word)
            else:
                new_word_list += [word]
        word_list += new_word_list
        relation_list.append(concat_words(new_word_list))
    return word_list+relation_list

def clean_relations(relation_list, glove_vocabulary):
    cleaned_relations = []
    for relation in relation_list:
        cleaned_relations.append(split_relation_into_words(relation, glove_vocabulary))
    return cleaned_relations

def build_vocabulary_embedding(relation_list, all_samples, glove_embedding,
                               embedding_size, dataset='fewrel'):
    vocabulary = {}
    embedding = []
    index = 0
    np.random.seed(100)
    # 0 as [pad]
    vocabulary['[pad]'] = index
    embedding.append(np.random.rand(embedding_size))
    index += 1

    # Encode words in relation label
    for relation in relation_list:
        for word in relation:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
                # init the word that are not in glove vocabulary randomly
                if word in glove_embedding:
                    embedding.append(glove_embedding[word])
                else:
                    embedding.append(np.random.rand(embedding_size))  # random embedding for unknown

    # encode sentence
    for sample in all_samples:
        question = sample[2]
        for word in question:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
                # init the word that are not in glove vocabulary randomly
                if word in glove_embedding:
                    embedding.append(glove_embedding[word])
                else:
                    #print(word)
                    embedding.append(np.random.rand(embedding_size))
        if dataset == 'fewrel':
            head = sample[3]
            if head not in vocabulary:
                vocabulary[head] = index
                index += 1
                if head in glove_embedding:
                    embedding.append(glove_embedding[head])
                else:
                    #print(word)
                    embedding.append(np.random.rand(embedding_size))

            tail = sample[5]
            if tail not in vocabulary:
                vocabulary[tail] = index
                index += 1
                if tail in glove_embedding:
                    embedding.append(glove_embedding[tail])
                else:
                    #print(word)
                    embedding.append(np.random.rand(embedding_size))

    return vocabulary, embedding

def words2indexs(word_list, vocabulary):
    index_list = []
    for word in word_list:
        index_list.append(vocabulary[word])
    return index_list

def transform_relations(relation_list, vocabulary):
    relation_ixs = []
    for relation in relation_list:
        relation_ixs.append(words2indexs(relation, vocabulary))
    return relation_ixs

# transform the words in the questions into index of the vocabulary
def transform_questions(sample_list, vocabulary, dataset):
    for sample in sample_list:
        sample[2] = words2indexs(sample[2], vocabulary)
        if dataset == 'fewrel':
            sample[3] = vocabulary[sample[3]]
            sample[5] = vocabulary[sample[5]]

    return sample_list

def generate_data(train_file, valid_file, test_file, relation_file, glove_file, embedding_size=300, dataset='fewrel'):
    train_data_list, train_data_dict = read_data(train_file)
    valid_data_list, valid_data_dict = read_data(valid_file)
    test_data_list, test_data_dict = read_data(test_file)
    relation_list, relation_dict = read_relation(relation_file)
    glove_vocabulary, glove_embedding = read_glove(glove_file)
    all_samples = train_data_list + valid_data_list + test_data_list

    cleaned_relations = clean_relations(relation_list, glove_vocabulary)

    vocabulary, embedding = build_vocabulary_embedding(cleaned_relations,
                                                       all_samples,
                                                       glove_embedding,
                                                       embedding_size,
                                                       dataset)

    relation_numbers = transform_relations(cleaned_relations, vocabulary)

    return train_data_list, train_data_dict, test_data_list, test_data_dict, valid_data_list, valid_data_dict, \
           relation_numbers, vocabulary, embedding

def random_split_relation(task_num, relation_dict):
    relation_num = len(relation_dict)
    relation_idx = [index + 1 for index in range(relation_num)]
    random.shuffle(relation_idx)
    rel2label = {}
    for rel_idx in relation_idx:
        rel2label[rel_idx] = relation_idx.index(rel_idx) % task_num

    return rel2label

def split_data(data, vocabulary, rel2cluster, task_num, instance_num=-1, dataset='fewrel'):  # -1 means all
    separated_data = [None] * task_num
    separated_relation = [None] * task_num
    for rel, items in data.items():
        rel_culter = rel2cluster[rel]
        items = transform_questions(items, vocabulary, dataset)
        if instance_num > 0:
            ins_num = instance_num if len(items) > instance_num else len(items)
            items = random.sample(items, ins_num)
            data[rel] = items

        if separated_data[rel_culter] is None:
            separated_data[rel_culter] = []
            separated_data[rel_culter].extend(items)
        else:
            separated_data[rel_culter].extend(items)

        if separated_relation[rel_culter] is None:
            separated_relation[rel_culter] = [rel]
        else:
            separated_relation[rel_culter].append(rel)

    return separated_data, separated_relation

def split_relation(relation):
    word_list = []
    # some relation will have fours parts, where the first part looks like
    # "base". We only choose the last three parts
    for word_seq in relation.split("/")[-3:]:
        for word in word_seq.split("_"):
            word_list += wordninja.split(word)
    return word_list

def get_embedding(relation_name, glove_embeddings):
    word_list = split_relation(relation_name)
    relation_embeddings = []
    for word in word_list:
        if word.lower() in glove_embeddings:
            relation_embeddings.append(glove_embeddings[word.lower()])
        else:
            print(word, "is not contained in glove vocabulary")
    return np.mean(relation_embeddings, 0)

def rel_glove_feature(relation_file, glove_file):
    relation_list, relation_dict = read_relation(relation_file)
    glove_vocabulary, glove_embedding = read_glove(glove_file)

    rel_glove_embedding = [None] * len(relation_dict)

    for rel, idx in relation_dict.items():
        rel_embedding = get_embedding(rel, glove_embedding)
        rel_glove_embedding[idx - 1] = rel_embedding  # rel_glove_embedding index from 0, relation_dict index from 1

    return rel_glove_embedding

def rel_kg_feature():
    pass

def cluster_data_by_glove(task_num, rel_features):
    cluster = KMeans(n_clusters=task_num).fit(rel_features)
    label = cluster.labels_
    rel2label = {}
    for index in range(len(label)):
        rel2label[index + 1] = label[index]
    # waits implement
    return rel2label

def read_relations_index(data_items):
    relation_pool = []
    for item in data_items:
        relation_number = item[0]
        if relation_number not in relation_pool:
            relation_pool.append(relation_number)
    return relation_pool

def read_relation_names(file_name, relation_index):
    all_relations = []
    with open(file_name) as in_file:
        for line in in_file:
            # remove "\n" for each line
            all_relations.append(line.split("\n")[0])
    relation_names = [all_relations[num-1] for num in relation_index]
    return relation_names

def read_glove_embeddings(glove_input_file):
    glove_dict = {}
    with open(glove_input_file) as in_file:
        for line in in_file:
            values = line.split()
            word = values[0]
            glove_dict[word] = np.asarray(values[1:], dtype='float32')
    return glove_dict

def gen_relation_embedding(train_data_list, valid_data_list, test_data_list, relation_names, glove_input_file):
    train_relation_index = read_relations_index(train_data_list)
    #print(train_relation_index)
    valid_relation_index = read_relations_index(valid_data_list)
    test_relation_index = read_relations_index(test_data_list)
    # Here list(a) will copy items in a. list.copy() not availabel in python2
    relation_index = list(train_relation_index)
    for index in test_relation_index+valid_relation_index:
        if index not in relation_index:
            relation_index.append(index)
    relation_index = np.array(relation_index)
    #print(relation_index[-1])
    relation_names = [relation_names[num-1] for num in relation_index]
    #vocabulary = gen_vocabulary(relation_names)
    glove_vocabulary, glove_embedding = read_glove(glove_input_file)
    #print(glove_embeddings)
    #print(glove_embeddings['dancer'])
    #print(vocabulary)
    #print(relation_names[-1])
    relation_embeddings = []
    for relation in relation_names:
        relation_embeddings.append(get_embedding(relation,
                                                 glove_embedding))

    relation_embeddings = np.asarray(relation_embeddings)
    '''
    relation_dict = {}
    for i in range(len(relation_index)):
        relation_dict[relation_index[i]] = i
    '''
    return relation_names, relation_index, relation_embeddings
    #print(len(relation_embeddings[0]))
    #np.save('relation_embeddings.npy', relation_embeddings)
    #relation_embeddings = np.load('relation_embeddings.npy')
    #print(len(relation_embeddings[0]))


def cluster_data(num_clusters, train_data_list, valid_data_list, test_data_list, relation_names, glove_input_file):
    relation_names, relation_index, relation_embeddings = \
        gen_relation_embedding(train_data_list, valid_data_list, test_data_list, relation_names, glove_input_file)
    kmeans = KMeans(n_clusters=num_clusters,
                    random_state=0).fit(relation_embeddings)
    #print(kmeans.inertia_)
    labels = kmeans.labels_
    rel_embed = {}
    cluster_index = {}
    for i in range(len(relation_index)):
        cluster_index[relation_index[i]] = labels[i]
        rel_embed[relation_index[i]] = relation_embeddings[i]
    rel_index = np.asarray(list(relation_index))

    return cluster_index, rel_embed

def random_split_data(num_clusters, train_data_list, valid_data_list, test_data_list, relation_names, glove_input_file):
    relation_names, relation_index, relation_embeddings = \
        gen_relation_embedding(train_data_list, valid_data_list, test_data_list, relation_names, glove_input_file)
    labels = {}
    random.shuffle(relation_index)
    for i in range(len(relation_index)):
        labels[relation_index[i]] = i % num_clusters
    rel_embed = {}
    cluster_index = {}
    for i in range(len(relation_index)):
        cluster_index[relation_index[i]] = labels[relation_index[i]]
        rel_embed[relation_index[i]] = relation_embeddings[i]
    rel_index = np.asarray(list(relation_index))
    return cluster_index, rel_embed

def load_data(train_file, valid_file, test_file, relation_file, glove_file, embedding_size=300,
              task_arrange='random', rel_encode='glove', task_num=10, instance_num=100, dataset='fewrel'):
    # generate data
    train_data_list, train_data_dict, test_data_list, test_data_dict, valid_data_list, valid_data_dict, \
    relation_numbers, vocabulary, embedding = \
        generate_data(train_file, valid_file, test_file, relation_file, glove_file, embedding_size, dataset)
    relation_list, relation_dict = read_relation(relation_file)

    # arrange task
    if task_arrange == 'random':
        # rel2cluster = random_split_relation(task_num, relation_dict)
        rel2cluster, rel_features = random_split_data(task_num, train_data_list, valid_data_list, test_data_list, relation_list, glove_file)
        # if rel_encode == 'glove':
        #     rel_features = rel_glove_feature(relation_file, glove_file)
        # elif rel_encode == 'kg':
        #     rel_features = rel_kg_feature()
        # else:
        #     raise Exception('rel_encode method %s not implement.' % rel_encode)

    elif task_arrange == 'cluster_by_glove_embedding':
        rel_features = rel_glove_feature(relation_file, glove_file)
        rel2cluster = cluster_data_by_glove(task_num, rel_features)

    elif task_arrange == 'origin':
        rel2cluster, rel_features = cluster_data(task_num, train_data_list, valid_data_list, test_data_list, relation_list, glove_file)
    else:
        raise Exception('task arrangement method %s not implement' % task_arrange)

    # -----------------------load tacred cluster and rel_features which same as EMAR------------------------------------
    if dataset == 'tacred':
        rel_cluster = np.load("dataset/tacred/rel_cluster_label.npy")
        rel2cluster = [i for i in rel_cluster]
        rel2cluster = {i+1: index for i, index in enumerate(rel2cluster)}
        rel_fea = np.load("dataset/tacred/rel_feature.npy")
        rel_fea = [i for i in rel_fea]
        rel_features = {i+1: fea for i, fea in enumerate(rel_fea)}
    # ---------------------------------------------------END------------------------------------------------------------

    split_train_data, split_train_relation = split_data(train_data_dict, vocabulary, rel2cluster, task_num, instance_num, dataset)
    split_test_data, split_test_relation = split_data(test_data_dict, vocabulary, rel2cluster, task_num, dataset=dataset)
    split_valid_data, split_valid_relation = split_data(valid_data_dict, vocabulary, rel2cluster, task_num, dataset=dataset)

    return split_train_data, train_data_dict, split_test_data, split_valid_data, relation_numbers, rel_features, \
           split_train_relation, vocabulary, embedding
