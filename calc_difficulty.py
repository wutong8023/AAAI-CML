import argparse
import json
from data_loader import *
from sklearn.metrics.pairwise import cosine_similarity

def compute_task_similarity(cos_sim, relation_split):
    task_num = len(relation_split)
    task_similarity = np.zeros([task_num, task_num])
    for current_task in range(task_num):
        for target_task in range(task_num):
            if current_task == target_task:
                continue

            current_task_rels = relation_split[current_task]
            target_task_rels = relation_split[target_task]

            task_sim = 0.0
            for rels in current_task_rels:
                for _rels in target_task_rels:
                    sim = cos_sim[rels - 1, _rels - 1]  # id -> index
                    task_sim += sim
            task_sim /= (len(current_task_rels) * len(target_task_rels))
            task_similarity[current_task, target_task] = task_sim

    return task_similarity

def main(opt):
    kl_rel_vec_file = './dataset/kl_rel_vec.json'
    with open(kl_rel_vec_file, 'r') as f:
        kl_rel_vec = json.load(f)

    split_train_data, train_data_dict, split_test_data, split_valid_data, relation_numbers, rel_features, \
    split_train_relations, vocabulary, embedding = \
        load_data(opt.train_file, opt.valid_file, opt.test_file, opt.relation_file, opt.glove_file,
                  opt.embedding_dim, opt.task_arrange, opt.rel_encode, opt.task_num,
                  opt.train_instance_num, opt.dataset)

    glove_rel_vec = [None] * 80
    for rel_id in rel_features:
        glove_rel_vec[rel_id - 1] = rel_features[rel_id]


    kl_rel_vec = np.array(kl_rel_vec[:80])
    glove_rel_vec = np.array(glove_rel_vec)

    kl_rel_cos_sim = cosine_similarity(kl_rel_vec)
    glove_rel_cos_sim = cosine_similarity(glove_rel_vec)

    # calc similarity between tasks
    kl_task_sim = compute_task_similarity(kl_rel_cos_sim, split_train_relations)
    glove_task_sim = compute_task_similarity(glove_rel_cos_sim, split_train_relations)

    kl_task_difficulty = np.sum(kl_task_sim, axis=1) / (len(split_train_relations) - 1)
    glove_task_difficulty = np.sum(glove_task_sim, axis=1) / (len(split_train_relations) - 1)

    print('\t'.join(['%.4f' % d for d in kl_task_difficulty]))
    print('\t'.join(['%.4f' % d for d in glove_task_difficulty]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cuda_id', default=0, type=int,
    #                     help='cuda device index, -1 means use cpu')

    parser.add_argument('--dataset', default='fewrel',
                        help='use which dataset')
    parser.add_argument('--train_file', default='dataset/training_data_with_entity.txt',
                        help='train file')
    parser.add_argument('--valid_file', default='dataset/test_data_with_entity.txt',
                        help='valid file')
    parser.add_argument('--test_file', default='dataset/test_data_with_entity.txt',
                        help='test file')
    parser.add_argument('--relation_file', default='dataset/relation_name.txt',
                        help='relation name file')
    # parser.add_argument('--dataset', default='simpleQuestion',
    #                     help='use which dataset')
    # parser.add_argument('--train_file', default='dataset/train_replace_ne.withpool',
    #                     help='train file')
    # parser.add_argument('--valid_file', default='dataset/valid_replace_ne.withpool',
    #                     help='valid file')
    # parser.add_argument('--test_file', default='dataset/test_replace_ne.withpool',
    #                     help='test file')
    # parser.add_argument('--relation_file', default='dataset/relation.2M.list',
    #                     help='relation name file')

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
    # parser.add_argument('--task_num', default=10, type=int,
    #                     help='number of tasks')
    parser.add_argument('--task_num', default=10, type=int,
                        help='number of tasks')
    parser.add_argument('--train_instance_num', default=200, type=int,
                        help='number of instances for one relation, -1 means all.')
    parser.add_argument('--loss_margin', default=0.5, type=float,
                        help='loss margin setting')
    parser.add_argument('--outside_epoch', default=300, type=float,
                        help='task level epoch')
    parser.add_argument('--early_stop', default=20, type=float,
                        help='task level epoch')
    parser.add_argument('--step_size', default=0.4, type=float,
                        help='step size Epsilon')
    parser.add_argument('--outer_step_formula', default='fixed', type=str,
                        help='outer step formula, fixed, linear, square_root')
    parser.add_argument('--learning_rate', default=2e-3, type=float,
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
    parser.add_argument('--sequence_index', default=0, type=int,
                        help='sequence index of tasks')

    opt = parser.parse_args()

    main(opt)