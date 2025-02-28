import torch.nn as nn

class RNN_nlp(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    """

    def __init__(self, nonlinearity, ntoken, ninp, nhid, nlayer, dropout=0.5, tie_weights=False):
        super(RNN_nlp, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.RNN(ninp, nhid, nlayer, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self._init_weights()

        self.nonlinearity = nonlinearity
        self.nhid = nhid
        self.nlayers = nlayer

    def _init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        """
        Args:
            - input: tensor(stride, bsz)
            - hidden: tnssor(nlays, bsz, ninp)
        """
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return weight.new_zeros(self.nlayers, bsz, self.nhid)