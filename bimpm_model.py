import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiMPM(nn.Module):
    """BiMPM model"""

    def __init__(self, vocab_size=0, embed_dim=10, weight_matrix=None, hidden_size=100, K=2):
        super(BiMPM, self).__init__()
        # no. of perspectives
        self.K = K
        self.hidden_size = hidden_size
        self.match_dim = K * 2
        # Word Representation Layer
        if weight_matrix is not None:
            self.embed_layer, self.num_embed, self.embed_dim = BiMPM._create_emb_layer(
                weight_matrix, is_trainable=False)
        else:
            self.num_embed, self.embed_dim = vocab_size, embed_dim
            self.embed_layer = nn.Embedding(vocab_size, embed_dim)

        # Context Representation Layer
        self.lstm_cxt_layer = nn.LSTM(self.embed_dim, hidden_size,
                                      1, batch_first=True, bidirectional=True)
        # multi-perspective matching operation
        self.W_full_f_12 = nn.Parameter(
            torch.randn(K, hidden_size), requires_grad=True)

        self.W_full_b_12 = nn.Parameter(
            torch.randn(K, hidden_size), requires_grad=True)

        self.W_full_f_21 = nn.Parameter(
            torch.randn(K, hidden_size), requires_grad=True)

        self.W_full_b_21 = nn.Parameter(
            torch.randn(K, hidden_size), requires_grad=True)

        # Aggregation Layer
        self.lstm_agg_layer = nn.LSTM(self.match_dim, hidden_size,
                                      1, batch_first=True, bidirectional=True)
        # prediction layer
        self.output_layer = nn.Sequential(
            nn.Linear(4 * hidden_size, 128),
            nn.Dropout(p=0.1),
            nn.Linear(128, 1),
        )

        self.dropout = nn.Dropout(p=0.1)

        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    @classmethod
    def _create_emb_layer(cls, weight_matrix, is_trainable=False):
        """
            Args:
                weight_matrix: FloatTensor of shape(num_embed, embed_dim)
        """
        num_embed, embed_dim = weight_matrix.size()
        emb_layer = nn.Embedding(num_embed, embed_dim)
        emb_layer.load_state_dict({'weight': weight_matrix})
        if is_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, num_embed, embed_dim

    def compute_full_cosine_match(self, q1s_vec, q2_Hn, W):

        scaled_q1s_vec = q1s_vec.unsqueeze(
            2) * (W.unsqueeze(0)).unsqueeze(0)  # NxTxKxd
        scaled_q2_Hn = q2_Hn.unsqueeze(1) * W.unsqueeze(0)  # NxKxd
        scaled_q2_Hn = scaled_q2_Hn.unsqueeze(1).repeat(
            1, scaled_q1s_vec.shape[1], 1, 1)  # NxT'(repeat)xKxd

        return F.cosine_similarity(scaled_q1s_vec, scaled_q2_Hn, dim=-1)

    def forward(self, q1s, q2s, q1_lens, q2_lens):
        # q1_mask = q1_mask.unsqueeze(-1)
        # q2_mask = q2_mask.unsqueeze(-1)
        q1_embedding = self.embed_layer(q1s)
        q2_embedding = self.embed_layer(q2s)

        #######
        # get the max sequence length
        q1_max_len, q2_max_len = q1s.size(1), q2s.size(1)
        packed_q1s = pack_padded_sequence(q1_embedding, q1_lens,
                                          batch_first=True,
                                          enforce_sorted=False)
        packed_q2s = pack_padded_sequence(q2_embedding, q2_lens,
                                          batch_first=True,
                                          enforce_sorted=False)

        packed_q1_cxt, (h1_n, c1_n) = self.lstm_cxt_layer(packed_q1s)
        q1_cxt, _ = pad_packed_sequence(packed_q1_cxt, padding_value=0.0, batch_first=True,
                                        total_length=q1_max_len)

        packed_q2_cxt, (h2_n, c2_n) = self.lstm_cxt_layer(packed_q2s)
        q2_cxt, _ = pad_packed_sequence(packed_q2_cxt, batch_first=True, padding_value=0.0,
                                        total_length=q2_max_len)

        q1_cxt = q1_cxt.view(q1_cxt.shape[0], q1_cxt.shape[1], 2, self.hidden_size)
        q1_cxt_f = q1_cxt[:, :, 0, :]
        q1_cxt_b = q1_cxt[:, :, 1, :]

        q2_cxt = q2_cxt.view(q2_cxt.shape[0], q2_cxt.shape[1], 2, self.hidden_size)
        q2_cxt_f = q2_cxt[:, :, 0, :]
        q2_cxt_b = q2_cxt[:, :, 1, :]

        h1_nf, h1_nb = h1_n[0], h1_n[1]
        h2_nf, h2_nb = h2_n[0], h2_n[1]

        # compute Full-Matching cosine similarity q1 to q2
        # forward direction
        full_match_12_f = self.compute_full_cosine_match(
            q1_cxt_f, h2_nf, self.W_full_f_12)
        # backward direction
        full_match_12_b = self.compute_full_cosine_match(
            q1_cxt_b, h2_nb, self.W_full_b_12)

        combined_match_vec_12 = torch.cat((full_match_12_f, full_match_12_b), dim=-1)

        # compute Full-Matching cosine similarity q2 to q1
        full_match_21_f = self.compute_full_cosine_match(
            q2_cxt_f, h1_nf, self.W_full_f_21)
        # backward direction
        full_match_21_b = self.compute_full_cosine_match(
            q2_cxt_b, h1_nb, self.W_full_b_21)

        combined_match_vec_21 = torch.cat((full_match_21_f, full_match_21_b), dim=-1)

        # agrregation of match vectors

        out12_agg, (hn_agg_12, _) = self.lstm_agg_layer(combined_match_vec_12)
        out21_agg, (hn_agg_21, _) = self.lstm_agg_layer(combined_match_vec_21)

        agg_match_vec = torch.cat(
            (hn_agg_12[0], hn_agg_12[1], hn_agg_21[0], hn_agg_21[1]), dim=-1)

        logits = self.output_layer(agg_match_vec)

        return logits.squeeze()
