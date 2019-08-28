import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import d

class GRU(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, emb, depth, hidden_size, seq_length, num_tokens, num_classes,
     ff_hidden_mult=2, dropout=0.0, directions=1):
        """
        :param emb: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_tokens: Number of tokens (usually words) in the vocabulary
        :param num_classes: Number of classes.
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        """
        super().__init__()

        self.num_tokens, self.hidden_size = num_tokens, hidden_size

        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        # self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        self.gru = nn.GRU(emb, self.hidden_size, depth, batch_first=True,
         dropout=dropout, bidirectional=(directions==2))

        self.ff = nn.Sequential(
            nn.Linear(directions*self.hidden_size, ff_hidden_mult * directions*self.hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * directions*self.hidden_size, num_classes)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        x, lens = x
        x = self.token_embedding(x)
        b, t, e = x.size()
        x = self.do(x)

        hs, hn = self.gru(x)

        # extract hidden at last non-pad token
        x = hs[torch.arange(b, device=d()), torch.clamp(lens-1, max=t-1), :]
        x = self.ff(x)

        return F.log_softmax(x, dim=1)