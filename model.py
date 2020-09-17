'''
This code is based on the Pytorch Orientaion:
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py
Original Author: Robert Guthrie
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class BiLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, vocab_embedding,
                 batch_size, device):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size


        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True).to(device)

        #self.maxpool = nn.MaxPool1d(hidden_dim*2)
        self.hidden = self.build_hidden()

    def build_hidden(self, batch_size = 1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #print(batch_size)
        return [torch.zeros(2, batch_size, self.hidden_dim),
                torch.zeros(2, batch_size, self.hidden_dim)]

    def init_hidden(self, device='cpu', batch_size = 1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #print(batch_size)
        self.hidden = (torch.zeros(2, batch_size, self.hidden_dim).to(device),
                       torch.zeros(2, batch_size, self.hidden_dim).to(device))
        #self.hidden[0] = self.hidden[0].to(device)
        #self.hidden[1] = self.hidden[1].to(device)

    def forward(self, packed_embeds):
        # 句子和关系的编码方法
        #print(packed_embeds)
        #print(self.hidden)
        lstm_out, self.hidden = self.lstm(packed_embeds, self.hidden)
        #maxpool_hidden = self.maxpool(lstm_out.view(1,len(sentence), -1))
        permuted_hidden = self.hidden[0].permute([1,0,2]).contiguous()
        #print(permuted_hidden.size())
        return permuted_hidden.view(-1, self.hidden_dim*2)

class SimilarityModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, vocab_embedding,
                 batch_size, device):
        super(SimilarityModel, self).__init__()
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(vocab_embedding))
        self.word_embeddings = self.word_embeddings.to(device)
        self.word_embeddings.weight.requires_grad = False
        self.sentence_biLstm = BiLSTM(embedding_dim, hidden_dim, vocab_size,
                                      vocab_embedding, batch_size, device)
        self.relation_biLstm = BiLSTM(embedding_dim, hidden_dim, vocab_size,
                                      vocab_embedding, batch_size, device)

    def init_hidden(self, device, batch_size=1):
        self.sentence_biLstm.init_hidden(device, batch_size)
        self.relation_biLstm.init_hidden(device, batch_size)

    def init_embedding(self, vocab_embedding):
        #print(self.word_embeddings(torch.tensor([27]).cuda()))
        self.word_embeddings.weight.data.copy_(torch.from_numpy(vocab_embedding))
        #print(self.word_embeddings(torch.tensor([27]).cuda()))

    def ranking_sequence(self, sequence):
        word_lengths = torch.tensor([len(sentence) for sentence in sequence])
        rankedi_word, indexs = word_lengths.sort(descending = True)
        ranked_indexs, inverse_indexs = indexs.sort()
        #print(indexs)
        sequence = [sequence[i] for i in indexs]
        return sequence, inverse_indexs

    def compute_que_embed(self, question_list, question_lengths,
                          reverse_question_indexs, reverse_model,
                          before_reverse=False):
        question_embeds = self.word_embeddings(question_list)
        question_packed = \
            torch.nn.utils.rnn.pack_padded_sequence(question_embeds,
                                                    question_lengths)
        question_embedding = self.sentence_biLstm(question_packed)
        question_embedding = question_embedding[reverse_question_indexs]
        if reverse_model is not None and not before_reverse:
            return reverse_model(question_embedding).detach()
        else:
            return question_embedding.detach()

    def compute_rel_embed(self, relation_list, relation_lengths,
                          reverse_relation_indexs, reverse_model,
                          before_reverse=False):
        relation_embeds = self.word_embeddings(relation_list)
        relation_packed = \
            torch.nn.utils.rnn.pack_padded_sequence(relation_embeds,
                                                    relation_lengths)
        relation_embedding = self.relation_biLstm(relation_packed)
        relation_embedding = relation_embedding[reverse_relation_indexs]
        if reverse_model is not None and not before_reverse:
            return reverse_model(relation_embedding).detach()
        else:
            return relation_embedding.detach()

    def forward(self, seq_list_1, seq_list_2, device,
                seq_list_1_idx, seq_list_2_idx,
                seq_list_1_lengths, seq_list_2_lengths, reverse_model=None, contrastive=False):
        # shape of question_list: (36, 128) 36 is the maximum length of questions, 128 is the batch size
        seq_list_1_embeds = self.word_embeddings(seq_list_1)
        seq_list_2_embeds = self.word_embeddings(seq_list_2)
        #print(question_lengths)
        seq_list_1_packed = \
            torch.nn.utils.rnn.pack_padded_sequence(seq_list_1_embeds,
                                                    seq_list_1_lengths)
        seq_list_2_packed = \
            torch.nn.utils.rnn.pack_padded_sequence(seq_list_2_embeds,
                                                    seq_list_2_lengths)
        seq_list_1_embedding = self.sentence_biLstm(seq_list_1_packed)  # shape
        if contrastive:
            seq_list_2_embedding = self.sentence_biLstm(seq_list_2_packed)
        else:
            seq_list_2_embedding = self.relation_biLstm(seq_list_2_packed)
        seq_list_1_embedding = seq_list_1_embedding[seq_list_1_idx]
        seq_list_2_embedding = seq_list_2_embedding[seq_list_2_idx]
        if reverse_model is not None:
            reverse_seq_list_1_embedding = reverse_model(seq_list_1_embedding)
            reverse_seq_list_2_embedding = reverse_model(seq_list_2_embedding)
            cos = nn.CosineSimilarity(dim=1)
            origin_score = cos(seq_list_1_embedding, seq_list_2_embedding)
            reverse_score = cos(reverse_seq_list_1_embedding,
                                 reverse_seq_list_2_embedding)
            avg_pooling = torch.mean(torch.stack((origin_score,
                                                  reverse_score)),0)
            return reverse_score
            #return max_pooling
        else:
            cos = nn.CosineSimilarity(dim=1)
            return cos(seq_list_1_embedding, seq_list_2_embedding)



class PCNN_Encoder(nn.Module):
    def __init__(self):
        super(PCNN_Encoder, self).__init__()

class PCNNModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, vocab_embedding,
                 batch_size, device, pos_limit=60, pos_dim=10):
        super(PCNNModel, self).__init__()
        # hyperparams

        # set glove embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(vocab_embedding))
        self.word_embeddings = self.word_embeddings.to(device)
        self.word_embeddings.weight.requires_grad = False

        # set pos embeddings
        # pos_size = 2 * pos_limit + 2, 0 for padding
        self.headPosEmbed = nn.Embedding(2 * pos_limit + 2, pos_dim, padding_idx=0)  # pos_size = max_length
        self.tailPosEmbed = nn.Embedding(2 * pos_limit + 2, pos_dim, padding_idx=0)

        self.conv = nn.Conv1d(embedding_dim + pos_dim * 2, 100, 3)
        self.pool = nn.MaxPool1d(120)
        # set mask embeddings
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.mask_embedding.weight.requires_grad = False
        self._minus = -100

        # relation encoder use BiLSTM
        self.relation_biLstm = BiLSTM(embedding_dim, hidden_dim, vocab_size,
                                      vocab_embedding, batch_size, device)

    def init_hidden(self, device, batch_size=1):
        pass

    def init_embedding(self, vocab_embedding):
        pass

    def ranking_sequence(self, sequence):
        pass

    def compute_que_embed(self, question_list, question_lengths,
                          reverse_question_indexs, reverse_model,
                          before_reverse=False):
        pass

    def compute_rel_embed(self, relation_list, relation_lengths,
                          reverse_relation_indexs, reverse_model,
                          before_reverse=False):
        pass

    def forward(self, question_list, relation_list, device,
                reverse_question_indexs, reverse_relation_indexs,
                question_lengths, relation_lengths, reverse_model=None):
        pass