import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LinearSST(nn.Module):
    def __init__(self, embedding_dim, granularity):
        super(LinearSST, self).__init__()

        self.linear = nn.Linear(embedding_dim, granularity)

    def forward(self, embedding):
        label_space = self.linear(embedding)
        label_scores = F.log_softmax(label_space)

        return label_scores
