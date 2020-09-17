import math
import pickle
import os
import sys
import time
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import scipy.stats as st

from utils import *
from data_loader import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import random
from model import SimilarityModel
from copy import deepcopy
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def select_neg_relations(target_rel, neg_rel_num, candidate_rel_list, current_model):
    pass


def feed_samples(model, samples, loss_function, all_relations, device, all_seen_relations=None):
    """
    :param model: SimilarityModel
    :param samples: batch samples
    :param loss_function: MarginLoss
    :param all_relations: word list for all relations [[rel_0_word_indices], [rel_1_word_indices], ..., [rel_80_word_indices]]
    :param device:
    :return:
    """
    questions, relations, relation_set_lengths = process_samples(
        samples, all_relations, device)  # expand samples
    ranked_questions, alignment_question_indexs = \
        ranking_sequence(questions)
    ranked_relations, alignment_relation_indexs = \
        ranking_sequence(relations)
    question_lengths = [len(question) for question in ranked_questions]
    relation_lengths = [len(relation) for relation in ranked_relations]
    pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)
    pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
    pad_questions = pad_questions.to(device)
    pad_relations = pad_relations.to(device)

    model.zero_grad()

    model.init_hidden(device, sum(relation_set_lengths))
    all_scores = model(pad_questions, pad_relations, device,
                       alignment_question_indexs, alignment_relation_indexs,
                       question_lengths, relation_lengths, None)
    all_scores = all_scores.to('cpu')
    pos_scores = []
    neg_scores = []
    pos_index = []
    start_index = 0
    for length in relation_set_lengths:
        pos_index.append(start_index)
        pos_scores.append(all_scores[start_index].expand(length - 1))
        neg_scores.append(all_scores[start_index + 1:start_index + length])
        start_index += length
    pos_scores = torch.cat(pos_scores)
    neg_scores = torch.cat(neg_scores)
    alignment_model_criterion = nn.MSELoss()

    loss = loss_function(pos_scores, neg_scores,
                         torch.ones(sum(relation_set_lengths) -
                                    len(relation_set_lengths)))
    loss.backward()
    return all_scores, loss


def evaluate_model(model, testing_data, batch_size, all_relations, device,
                   reverse_model=None):
    """
    :param model:
    :param testing_data:
    :param batch_size:
    :param all_relations:
    :param device:
    :param reverse_model:
    :return:
    """
    num_correct = 0
    for i in range((len(testing_data) - 1) // batch_size + 1):
        samples = testing_data[i * batch_size:(i + 1) * batch_size]
        gold_relation_indexs, questions, relations, relation_set_lengths = \
            process_testing_samples(samples, all_relations, device)
        model.init_hidden(device, sum(relation_set_lengths))
        ranked_questions, reverse_question_indexs = \
            ranking_sequence(questions)
        ranked_relations, reverse_relation_indexs = \
            ranking_sequence(relations)
        question_lengths = [len(question) for question in ranked_questions]
        relation_lengths = [len(relation) for relation in ranked_relations]
        pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)
        pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
        all_scores = model(pad_questions, pad_relations, device,
                           reverse_question_indexs, reverse_relation_indexs,
                           question_lengths, relation_lengths, reverse_model)
        start_index = 0
        for j in range(len(relation_set_lengths)):
            length = relation_set_lengths[j]
            cand_indexs = samples[j][1]
            pred_index = (cand_indexs[
                all_scores[start_index:start_index + length].argmax()])
            if pred_index == gold_relation_indexs[j]:
                num_correct += 1
            start_index += length
    return float(num_correct) / len(testing_data)


def print_list(result):
    for num in result:
        sys.stdout.write('%.3f, ' % num)
    print('')


def interval(data):
    """
    data: 1-dim np array
    """
    interv = st.t.interval(0.95, len(data) - 1, loc=np.mean(data), scale=st.sem(data))
    mean = np.mean(data)
    interv = interv - mean
    return mean, interv
# -----------------------------


def update_rel_cands(memory_data, all_seen_cands, num_cands):
    if len(memory_data) > 0:
        for this_memory in memory_data:
            for sample in this_memory:
                valid_rels = [rel for rel in all_seen_cands if rel != sample[0]]
                sample[1] = random.sample(valid_rels, min(num_cands, len(valid_rels)))


def offset_list(l, offset):
    if offset == 0:
        return l
    offset_l = [None] * len(l)
    for i in range(len(l)):
        offset_l[(i + offset) % len(l)] = l[i]

    return offset_l


def resort_list(l, index):
    resorted_l = [None] * len(l)
    for i in range(len(index)):
        resorted_l[i] = l[index[i]]

    return resorted_l


def resort_memory(memory_pool, similarity_index):
    memory_pool = sorted(memory_pool, key=lambda item: np.argwhere(similarity_index == item[0]))
    return memory_pool


# get relation embedding of current seen relations
def tsne_relations(model, seen_task_relations, all_relations, device, task_idx, alignment_model=None,
                   before_alignment=False):
    color_schema = ['black', 'darkviolet', 'firebrick', 'green', 'gold',
                    'chartreuse', 'darkorange', 'chocolate', 'cyan', 'grey']
    task_labels = ['Task %d' % idx for idx in task_idx]
    # get relation embeddings of current seen relations
    current_seen_relations = []
    relation_cluster = []
    for i in range(len(seen_task_relations)):
        current_seen_relations.extend(seen_task_relations[i])
        relation_cluster.extend([i] * len(seen_task_relations[i]))

    relations_index = []
    for rel in current_seen_relations:
        relations_index.append(torch.tensor(all_relations[rel - 1], dtype=torch.long).to(device))

    model.init_hidden(device, len(relations_index))
    ranked_relations, alignment_relation_indexs = ranking_sequence(relations_index)
    relation_lengths = [len(relation) for relation in ranked_relations]

    pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)

    rel_embeds = model.compute_rel_embed(pad_relations, relation_lengths,
                                         alignment_relation_indexs,
                                         alignment_model, before_alignment)
    rel_embeds = rel_embeds.detach().cpu().numpy()

    # # draw tsne picture
    # X_tsne = TSNE(n_components=2, random_state=33).fit_transform(rel_embeds)
    # task_label_cords_list = [None] * len(seen_task_relations)
    # for i in range(len(current_seen_relations)):
    #     relation_idx = current_seen_relations[i]
    #     rel_cluster = relation_cluster[i]
    #     relation_cord = X_tsne[i]
    #     relation_color = color_schema[rel_cluster]
    #     plt.scatter(relation_cord[0], relation_cord[1], alpha=0.6, marker='o', c=relation_color)
    #     # plt.text(relation_cord[0], relation_cord[1] + 1.0, str(relation_idx), c=relation_color)
    #
    #     if task_label_cords_list[rel_cluster] is None:
    #         task_label_cords_list[rel_cluster] = [relation_cord]
    #     else:
    #         task_label_cords_list[rel_cluster].append(relation_cord)
    #
    # # add task label
    # for i in range(len(task_label_cords_list)):
    #     task_label_cords = task_label_cords_list[i]
    #     task_label_cord = np.mean(np.array(task_label_cords), axis=0)
    #     plt.text(task_label_cord[0], task_label_cord[1] + 2.0, task_labels[i], c=color_schema[i])
    #
    # plt.title('Relation embedding distance t-SNE plot after %d tasks trained' % len(seen_task_relations),
    #           fontsize='large', fontweight='bold', color='black')
    # plt.show()

    return rel_embeds


def main(opt):
    print(opt)
    random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    np.random.RandomState(opt.random_seed)
    start_time = time.time()
    checkpoint_dir = os.path.join(opt.checkpoint_dir, '%.f' % start_time)

    device = torch.device(('cuda:%d' % opt.cuda_id) if torch.cuda.is_available() and opt.cuda_id >= 0 else 'cpu')

    # do following process
    split_train_data, train_data_dict, split_test_data, split_valid_data, relation_numbers, rel_features, \
    split_train_relations, vocabulary, embedding = \
        load_data(opt.train_file, opt.valid_file, opt.test_file, opt.relation_file, opt.glove_file,
                  opt.embedding_dim, opt.task_arrange, opt.rel_encode, opt.task_num,
                  opt.train_instance_num, opt.dataset)
    print(split_train_relations)

    # ------------------------------------------------------------------------
    # save cluster results
    our_tasks = split_train_relations

    count = 0
    for i in our_tasks:
        count += len(i)

    portion = np.zeros(10000)
    portion = portion-1
    for i in range(len(our_tasks)):
        for j in our_tasks[i]:
            portion[j - 1] = int(i)
    np.save("dataset/tacred/CML_tacred_random.npy", np.array(portion).astype(int))
    # -------------------------------------------------------------------------

    print('\n'.join(
        ['Task %d\t%s' % (index, ', '.join(['%d' % rel for rel in split_train_relations[index]])) for index in
         range(len(split_train_relations))]))

    task_sequence = list(range(opt.task_num))
    if opt.random_idx:
        for i in range(opt.random_times):
            random.shuffle(task_sequence)

    offset_seq = task_sequence[-opt.sequence_index:] + task_sequence[:-opt.sequence_index]

    split_train_data = resort_list(split_train_data, offset_seq)
    split_test_data = resort_list(split_test_data, offset_seq)
    split_valid_data = resort_list(split_valid_data, offset_seq)
    split_train_relations = resort_list(split_train_relations, offset_seq)
    print('[%s]' % ', '.join(['Task %d' % idx for idx in offset_seq]))

    relid2embedidx = {}
    embedidx2relid = {}
    if opt.similarity == 'kl_similarity':
        kl_dist_ht = read_json(opt.kl_dist_file)

        sorted_similarity_index = np.argsort(np.asarray(kl_dist_ht), axis=1) + 1
    elif opt.similarity == 'glove_similarity':
        glove_embedding = []

        embed_id = 0
        for rel_id in rel_features:
            glove_embedding.append(rel_features[rel_id])
            relid2embedidx[rel_id] = embed_id
            embedidx2relid[embed_id] = rel_id
            embed_id += 1

        glove_similarity = cosine_similarity(np.asarray(glove_embedding))
        glove_dist = np.sqrt(1 - np.power(np.where(glove_similarity > 1.0, 1.0, glove_similarity), 2))
        sorted_embed_index = np.argsort(np.asarray(glove_dist), axis=1)
        sorted_similarity_index = np.zeros(sorted_embed_index.shape)
        for i in range(sorted_embed_index.shape[0]):
            for j in range(sorted_embed_index.shape[1]):
                sorted_similarity_index[i][j] = embedidx2relid[sorted_embed_index[i][j]]
    else:
        raise Exception('similarity method not implemented')

    # prepare model
    inner_model = SimilarityModel(opt.embedding_dim, opt.hidden_dim, len(vocabulary),
                                  np.array(embedding), 1, device)

    memory_data = []
    memory_pool = []
    memory_question_embed = []
    memory_relation_embed = []
    sequence_results = []
    result_whole_test = []
    seen_relations = []
    all_seen_relations = []
    rel2instance_memory = {}
    memory_index = 0
    seen_task_relations = []
    rel_embeddings = []
    for task_ix in range(opt.task_num):  # outside loop
        # reptile start model parameters pi
        weights_before = deepcopy(inner_model.state_dict())

        train_task = split_train_data[task_ix]
        test_task = split_test_data[task_ix]
        valid_task = split_valid_data[task_ix]
        train_relations = split_train_relations[task_ix]
        seen_task_relations.append(train_relations)

        # collect seen relations
        for data_item in train_task:
            if data_item[0] not in seen_relations:
                seen_relations.append(data_item[0])

        # remove unseen relations
        current_train_data = remove_unseen_relation(train_task, seen_relations, dataset=opt.dataset)
        current_valid_data = remove_unseen_relation(valid_task, seen_relations, dataset=opt.dataset)

        current_test_data = []
        for previous_task_id in range(task_ix + 1):
            current_test_data.append(
                remove_unseen_relation(split_test_data[previous_task_id], seen_relations, dataset=opt.dataset))

        for this_sample in current_train_data:
            if this_sample[0] not in all_seen_relations:
                all_seen_relations.append(this_sample[0])

        update_rel_cands(memory_data, all_seen_relations, opt.num_cands)

        # train inner_model
        loss_function = nn.MarginRankingLoss(opt.loss_margin)
        inner_model = inner_model.to(device)
        optimizer = optim.Adam(inner_model.parameters(), lr=opt.learning_rate)
        t = tqdm(range(opt.outside_epoch))
        best_valid_acc = 0.0
        early_stop = 0
        best_checkpoint = ''

        #
        resorted_memory_pool = []
        for epoch in t:
            batch_num = (len(current_train_data) - 1) // opt.batch_size + 1
            total_loss = 0.0
            target_rel = -1
            for batch in range(batch_num):

                batch_train_data = current_train_data[batch * opt.batch_size: (batch + 1) * opt.batch_size]

                if len(memory_data) > 0:
                    # CML
                    if target_rel == -1 or len(resorted_memory_pool) == 0:
                        target_rel = batch_train_data[0][0]
                        if opt.similarity == 'kl_similarity':
                            target_rel_sorted_index = sorted_similarity_index[target_rel - 1]
                        else:
                            target_rel_sorted_index = sorted_similarity_index[relid2embedidx[target_rel]]
                        resorted_memory_pool = resort_memory(memory_pool, target_rel_sorted_index)

                    if len(resorted_memory_pool) >= opt.task_memory_size:
                        current_memory = resorted_memory_pool[:opt.task_memory_size]
                        resorted_memory_pool = resorted_memory_pool[opt.task_memory_size + 1:]  # update rest memory
                        batch_train_data.extend(current_memory)
                    else:
                        current_memory = resorted_memory_pool[:]
                        resorted_memory_pool = []
                        batch_train_data.extend(current_memory)

                    # MLLRE
                    # all_seen_data = []
                    # for one_batch_memory in memory_data:
                    #     all_seen_data += one_batch_memory
                    #
                    # memory_batch = memory_data[memory_index]
                    # batch_train_data.extend(memory_batch)
                    # scores, loss = feed_samples(inner_model, memory_batch, loss_function, relation_numbers, device)
                    # optimizer.step()
                    # memory_index = (memory_index+1) % len(memory_data)

                if len(rel2instance_memory) > 0:  # from the second task, this will not be empty
                    if opt.is_curriculum_train == 'Y':
                        current_train_rel = batch_train_data[0][0]
                        current_rel_similarity_sorted_index = sorted_similarity_index[current_train_rel + 1]
                        seen_relation_sorted_index = []
                        for rel in current_rel_similarity_sorted_index:
                            if rel in rel2instance_memory.keys():
                                seen_relation_sorted_index.append(rel)

                        curriculum_rel_list = []
                        if opt.sampled_rel_num >= len(seen_relation_sorted_index):
                            curriculum_rel_list = seen_relation_sorted_index[:]
                        else:
                            step = len(seen_relation_sorted_index) // opt.sampled_rel_num
                            for i in range(0, len(seen_relation_sorted_index), step):
                                curriculum_rel_list.append(seen_relation_sorted_index[i])

                        # curriculum select relation
                        instance_list = []
                        for sampled_relation in curriculum_rel_list:
                            if opt.mini_batch_split == 'Y':
                                instance_list.append(rel2instance_memory[sampled_relation])
                            else:
                                instance_list.extend(rel2instance_memory[sampled_relation])
                    else:
                        # randomly select relation
                        instance_list = []
                        random_relation_list = random.sample(list(rel2instance_memory.keys()),
                                                             min(opt.sampled_rel_num, len(rel2instance_memory)))
                        for sampled_relation in random_relation_list:
                            if opt.mini_batch_split == 'Y':
                                instance_list.append(rel2instance_memory[sampled_relation])
                            else:
                                instance_list.extend(rel2instance_memory[sampled_relation])

                    if opt.mini_batch_split == 'Y':
                        for one_batch_instance in instance_list:
                            scores, loss = feed_samples(inner_model, one_batch_instance, loss_function,
                                                        relation_numbers, device, all_seen_relations)
                            optimizer.step()
                    else:
                        scores, loss = feed_samples(inner_model, instance_list, loss_function, relation_numbers, device,
                                                    all_seen_relations)
                        optimizer.step()

                scores, loss = feed_samples(inner_model, batch_train_data, loss_function, relation_numbers, device,
                                            all_seen_relations)
                optimizer.step()
                total_loss += loss

            # valid test
            valid_acc = evaluate_model(inner_model, current_valid_data, opt.batch_size, relation_numbers, device)
            # checkpoint
            checkpoint = {'net_state': inner_model.state_dict(), 'optimizer': optimizer.state_dict()}
            if valid_acc > best_valid_acc:
                best_checkpoint = '%s/checkpoint_task%d_epoch%d.pth.tar' % (checkpoint_dir, task_ix + 1, epoch)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                torch.save(checkpoint, best_checkpoint)
                best_valid_acc = valid_acc
                early_stop = 0
            else:
                early_stop += 1

            # print()
            t.set_description('Task %i Epoch %i' % (task_ix + 1, epoch + 1))
            t.set_postfix(loss=total_loss.item(), valid_acc=valid_acc, early_stop=early_stop,
                          best_checkpoint=best_checkpoint)
            t.update(1)

            if early_stop >= opt.early_stop and task_ix != 0:
                # convergence
                break

            if task_ix == 0 and early_stop >= 40:
                break
        t.close()
        print('Load best check point from %s' % best_checkpoint)
        checkpoint = torch.load(best_checkpoint)

        weights_after = checkpoint['net_state']
        if opt.outer_step_formula == 'fixed':
            outer_step_size = opt.step_size
        elif opt.outer_step_formula == 'linear':
            outer_step_size = opt.step_size * (1 - task_ix / opt.task_num)
        elif opt.outer_step_formula == 'square_root':
            outer_step_size = math.sqrt(opt.step_size * (1 - task_ix / opt.task_num))
        # outer_step_size = 0.4
        inner_model.load_state_dict(
            {name: weights_before[name] + (weights_after[name] - weights_before[name]) * outer_step_size
             for name in weights_before})

        results = [evaluate_model(inner_model, test_data, opt.batch_size, relation_numbers, device)
                   for test_data in current_test_data]

        # sample memory from current_train_data
        if opt.memory_select_method == 'select_for_relation':
            # sample instance for one relation
            for rel in train_relations:
                rel_items = remove_unseen_relation(train_data_dict[rel], seen_relations, dataset=opt.dataset)
                rel_memo = select_data(inner_model, rel_items, int(opt.sampled_instance_num),
                                       relation_numbers, opt.batch_size, device)
                rel2instance_memory[rel] = rel_memo

        if opt.memory_select_method == 'select_for_task':
            # sample instance for one Task
            rel_instance_num = math.ceil(opt.sampled_instance_num_total / len(train_relations))
            for rel in train_relations:
                rel_items = remove_unseen_relation(train_data_dict[rel], seen_relations, dataset=opt.dataset)
                rel_memo = select_data(inner_model, rel_items, rel_instance_num,
                                       relation_numbers, opt.batch_size, device)
                rel2instance_memory[rel] = rel_memo

        if opt.task_memory_size > 0:
            # sample memory from current_train_data
            if opt.memory_select_method == 'random':
                memory_data.append(random_select_data(current_train_data, int(opt.task_memory_size)))
            elif opt.memory_select_method == 'vec_cluster':
                selected_memo = select_data(inner_model, current_train_data, int(opt.task_memory_size),
                                            relation_numbers, opt.batch_size, device)
                memory_data.append(selected_memo)  # memorydata-list
                memory_pool.extend(selected_memo)
            elif opt.memory_select_method == 'difficulty':
                memory_data.append()

        print_list(results)
        avg_result = sum(results) / len(results)
        test_set_size = [len(testdata) for testdata in current_test_data]
        whole_result = sum([results[i] * test_set_size[i] for i in range(len(current_test_data))]) / sum(test_set_size)
        print('test_set_size: [%s]' % ', '.join([str(size) for size in test_set_size]))
        print('avg_acc: %.3f, whole_acc: %.3f' % (avg_result, whole_result))

    print('test_all:')
    result_total_for_avg = []
    result_total_for_whole = []
    for epoch in range(10):
        current_test_data = []
        for previous_task_id in range(opt.task_num):
            current_test_data.append(
                remove_unseen_relation(split_test_data[previous_task_id], seen_relations, dataset=opt.dataset))

        loss_function = nn.MarginRankingLoss(opt.loss_margin)
        optimizer = optim.Adam(inner_model.parameters(), lr=opt.learning_rate)
        optimizer.zero_grad()
        for one_batch_memory in memory_data:
            scores, loss = feed_samples(inner_model, one_batch_memory, loss_function, relation_numbers, device,
                                        all_seen_relations)
            optimizer.step()
        results = [evaluate_model(inner_model, test_data, opt.batch_size, relation_numbers, device)
                   for test_data in current_test_data]
        print(results)

        avg_result = sum(results) / len(results)
        test_set_size = [len(testdata) for testdata in current_test_data]
        whole_result = sum([results[i] * test_set_size[i] for i in range(len(current_test_data))]) / sum(test_set_size)

        print('test_set_size: [%s]' % ', '.join([str(size) for size in test_set_size]))
        print('avg_acc: %.3f, whole_acc: %.3f' % (avg_result, whole_result))
        result_total_for_avg.append(results)
        result_total_for_whole.append(whole_result)

    # clean saved parameters
    files = os.listdir(checkpoint_dir)
    for weigths_file in files:
        os.remove(os.path.join(checkpoint_dir, weigths_file))
    os.removedirs(checkpoint_dir)

    # -----------------------------------------------------------
    # 输出结果
    avg_total = np.mean(np.array(result_total_for_avg), 1)
    avg_mean, avg_interval = interval(avg_total)
    whole_mean, whole_interval = interval(np.array(result_total_for_whole))
    result_total = {"avg_acc": result_total_for_avg, "whole_acc": result_total_for_whole,
                    "avg_mean": avg_mean, "avg_interval": avg_interval.tolist(),
                    "whole_mean": whole_mean, "whole_interval": whole_interval.tolist()}
    print(result_total)

    with open(opt.result_file, "w") as file_out:
        json.dump(result_total, file_out)
        # json.dump(file_out, result_total)
    # -------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', default="dataset/tacred/mllre_tacred.json",
                        help="restore the results")
    parser.add_argument('--cuda_id', default=0, type=int,
                        help='cuda device index, -1 means use cpu')

    # SimpleQuestion
    parser.add_argument('--task_num', default=20, type=int,
                        help='number of tasks')
    parser.add_argument('--dataset', default='simpleQuestion',
                        help='use which dataset')
    parser.add_argument('--train_file', default='dataset/train_replace_ne.withpool',
                        help='train file')
    parser.add_argument('--valid_file', default='dataset/valid_replace_ne.withpool',
                        help='valid file')
    parser.add_argument('--test_file', default='dataset/test_replace_ne.withpool',
                        help='test file')
    parser.add_argument('--relation_file', default='dataset/relation.2M.list',
                        help='relation name file')

    parser.add_argument('--glove_file', default='dataset/glove.6B.300d.txt',
                        help='glove embedding file')
    parser.add_argument('--embedding_dim', default=300, type=int,
                        help='word embeddings dimensional')
    parser.add_argument('--hidden_dim', default=200, type=int,
                        help='BiLSTM hidden dimensional')
    parser.add_argument('--task_arrange', default='origin',
                        help='task arrangement method, e.g. origin, cluster_by_glove_embedding, random')
    parser.add_argument('--rel_encode', default='glove',
                        help='relation encode method')
    parser.add_argument('--meta_method', default='reptile',
                        help='meta learning method, maml and reptile can be choose')
    parser.add_argument('--num_cands', default=10, type=int,
                        help='candidate negative relation numbers in memory')
    parser.add_argument('--batch_size', default=50, type=float,
                        help='Reptile inner loop batch size')
    # parser.add_argument('--task_num', default=20, type=int,
    #                     help='number of tasks')
    parser.add_argument('--train_instance_num', default=200, type=int,
                        help='number of instances for one relation, -1 means all.')
    parser.add_argument('--loss_margin', default=0.5, type=float,
                        help='loss margin setting')
    parser.add_argument('--outside_epoch', default=300, type=float,
                        help='task level epoch')
    parser.add_argument('--early_stop', default=20, type=float,
                        help='task level epoch')
    parser.add_argument('--step_size', default=0.5, type=float,
                        help='step size Epsilon')
    parser.add_argument('--outer_step_formula', default='fixed', type=str,
                        help='outer step formula, fixed, linear, square_root')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--random_seed', default=100, type=int,
                        help='random seed')
    parser.add_argument('--task_memory_size', default=50, type=int,
                        help='number of samples for each task')
    parser.add_argument('--memory_select_method', default='vec_cluster',
                        help='the method of sample memory data, e.g. vec_cluster, random, difficulty, select_for_relation, select_for_task')
    parser.add_argument('--is_curriculum_train', default='Y',
                        help='when training with memory, this will control if relations are curriculumly sampled.')
    parser.add_argument('--mini_batch_split', default='N',
                        help='whether mini-batch split into sampled_rel_num batches, Y or N')
    parser.add_argument('--checkpoint_dir', default='./checkpoint',
                        help='check point dir')
    parser.add_argument('--sampled_rel_num', default=10,
                        help='relation sampled number for current training relation')
    parser.add_argument('--sampled_instance_num', default=6,
                        help='instance sampled number for a sampled relation, total sampled 6 * 80 instances ')
    parser.add_argument('--sampled_instance_num_total', default=50,
                        help='instance sampled number for a task, total sampled 50 instances ')
    parser.add_argument('--similarity', default='glove_similarity',
                        help='the similarity calculate method, kl_similarity, glove_similarity')
    parser.add_argument('--kl_dist_file', default='dataset/kl_dist_ht.json',
                        help='glove embedding file')
    parser.add_argument('--random_idx', default=False, type=bool,
                        help='if corrupt task sequence')
    parser.add_argument('--random_times', default=1, type=int,
                        help='randomly corrupt task sequence times')
    parser.add_argument('--index', default=1, type=int,
                        help='experiment index')
    parser.add_argument('--sequence_index', default=10, type=int,
                        help='sequence index of tasks')
    parser.add_argument('--if_contrast_relation', default=False, type=bool,
                        help='if contrast relation with most similar relations')

    opt = parser.parse_args()

    main(opt)
