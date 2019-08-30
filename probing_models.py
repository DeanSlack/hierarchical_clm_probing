import torch.nn as nn
import torch.nn.functional as F


class LinearSST(nn.Module):
    def __init__(self, embedding_dim, granularity):
        super(LinearSST, self).__init__()

        self.linear = nn.Linear(embedding_dim, granularity)

    def forward(self, embedding):
        label_space = self.linear(embedding)
        label_scores = F.log_softmax(label_space, dim=1)

        return label_scores


class NonLinearSST(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, granularity):
        super(NonLinearSST, self).__init__()

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, granularity)

    def forward(self, embedding):
        hidden_space = F.relu(self.linear1(embedding))
        label_space = self.linear2(hidden_space)
        label_scores = F.log_softmax(label_space, dim=1)

        return label_scores


class LinearPOS(nn.Module):
    def __init__(self, embedding_dim, granularity):
        super(LinearPOS, self).__init__()

        self.linear = nn.Linear(embedding_dim, granularity)

    def forward(self, embedding):
        label_space = self.linear(embedding)
        label_scores = F.log_softmax(label_space, dim=1)

        return label_scores
