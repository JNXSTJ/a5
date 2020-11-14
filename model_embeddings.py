#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        # YOUR CODE HERE for part 1h
        self.dropout_rate = 0.3
        self.highway = Highway(word_embed_size)
        self.cnn = CNN(word_embed_size)
        self.vocab = vocab
        length = len(vocab)
        self.embeddings = nn.Embedding(length, word_embed_size)
        self.dropout = nn.Dropout(self.dropout_rate)

        # END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        # YOUR CODE HERE for part 1h
        # x_emb[sentence_length, batch_size, max_word_length, word_embed_size]
        x_emb = self.embeddings(input)
        # x_reshaped[sentence_length, batch_size, word_embed_size, max_word_length]
        x_reshaped = x_emb.transpose(2, 3)
        # x_conv[sentence_length, batch_size, word_embed_size, max_word_length - k + 1 + pad * 2]
        shape = x_reshaped.shape
        x_conv = self.cnn(x_reshaped.reshape([shape[0] * shape[1], shape[2], shape[3]]))
        x_conv = x_conv.reshape([shape[0], shape[1], shape[2], -1])
        x_convout = torch.max(torch.nn.functional.relu(x_conv), -1).values
        x_convout = torch.squeeze(x_convout, -1)
        x_highway = self.highway(x_convout)
        x_wordemb = self.dropout(x_highway)

        return x_wordemb
        # END YOUR CODE
