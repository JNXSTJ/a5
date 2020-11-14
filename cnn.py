#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    # YOUR CODE HERE for part 1g

    def __init__(self, e_word, kernel_size=5, padding=1):
        super(CNN, self).__init__()
        self.e_word = e_word
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = nn.Conv1d(e_word, e_word, kernel_size, padding=1)

    def forward(self, x_reshaped):
        """	Using convolutinoal network to convert x_reshaped to x_convout
                        @param x_reshaped (tensor) : x_reshaped[batch, e_word]
                        @return x_convout (tensor) : x_reshaped[batch, e_word]
        """

        x_convout = self.conv(x_reshaped)
        return x_convout

    # END YOUR CODE
