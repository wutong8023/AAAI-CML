import pickle
import random
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
random.seed(100)
np.random.seed(100)
np.random.RandomState(100)
color_schema = ['black', 'darkviolet', 'firebrick', 'green', 'gold',
                'chartreuse', 'darkorange', 'chocolate', 'cyan', 'grey']

curriculum_relation_embedding_file_path = './results/curriculum_rel_embeddings_offset8.pkl'
mllre_relation_embedding_file_path = './results/mllre_rel_embeddings_offset8.pkl'
kl_distance_file_path = './dataset/kl_dist_ht.json'

# read data
with open(mllre_relation_embedding_file_path, 'rb') as f:
    mllre_rel_embeddings, mllre_task_relations, mllre_task_index = pickle.load(f)

with open(curriculum_relation_embedding_file_path, 'rb') as f:
    curriculum_rel_embeddings, curriculum_task_relations, curriculum_task_index = pickle.load(f)

with open(kl_distance_file_path, 'r') as f:
    kl_dist = json.load(f)

# process data
mllre_relations_ids = []
mllre_relation_clusters = []
mllre_task_labels = ['Task %d' % idx for idx in mllre_task_index]
for i in range(len(mllre_task_relations)):
    mllre_relations_ids.extend(mllre_task_relations[i])
    mllre_relation_clusters.extend([i] * len(mllre_task_relations[i]))

# curriculum_relations_ids = []
# curriculum_relation_clusters = []
# curriculum_task_labels = ['Task %d' % idx for idx in curriculum_task_index]
# for i in range(len(curriculum_task_relations)):
#     curriculum_relations_ids.extend(curriculum_task_relations[i])
#     curriculum_relation_clusters.extend([i] * len(curriculum_task_relations[i]))

mllre_tsne = TSNE(n_components=2, random_state=33).fit_transform(mllre_rel_embeddings[-1])
# curriculum_tsne = TSNE(n_components=2, random_state=33).fit_transform(curriculum_rel_embeddings[-1])
# kl_tsne = TSNE(n_components=2, random_state=256, metric="precomputed").fit_transform(np.abs(np.asarray(kl_dist)))

#
print('draw picture')
# set size
plt.figure(figsize=(8, 5), dpi=600)
# plt.rcParams['figure.dpi'] = 600
# draw first plot
# plt.subplot(3, 1, 1)
# mllre_tsne
mllre_task_label_cords_list = [None] * len(mllre_task_relations)

for i in range(len(mllre_relations_ids)):
    relation_idx = mllre_relations_ids[i]
    rel_cluster = mllre_relation_clusters[i]
    relation_cord = mllre_tsne[i]
    relation_color = color_schema[rel_cluster]
    plt.scatter(relation_cord[0], relation_cord[1], alpha=0.6, marker='o', c=relation_color)
    # plt.text(relation_cord[0], relation_cord[1] + 1.0, str(relation_idx), c=relation_color)

    if mllre_task_label_cords_list[rel_cluster] is None:
        mllre_task_label_cords_list[rel_cluster] = [relation_cord]
    else:
        mllre_task_label_cords_list[rel_cluster].append(relation_cord)

# add task label
for i in range(len(mllre_task_label_cords_list)):
    task_label_cords = mllre_task_label_cords_list[i]
    task_label_cord = np.mean(np.array(task_label_cords), axis=0)
    plt.text(task_label_cord[0], task_label_cord[1], mllre_task_labels[i], c=color_schema[i])

plt.savefig('./results/relation_tsne.png', dpi=600)
plt.show()
