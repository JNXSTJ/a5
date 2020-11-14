#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    # YOUR CODE HERE for part 1f

    def __init__(self, eword):
        super(Highway, self).__init__()
        self.input_size = eword
        self.output_size = eword
        self.W_proj = nn.Linear(self.input_size, self.output_size)
        self.W_gate = nn.Linear(self.input_size, self.output_size)

    # x_convout[batch, eword]
    def forward(self, x_convout):
        x_proj = nn.functional.relu(self.W_proj(x_convout))
        x_gate = torch.sigmoid(self.W_gate(x_convout))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_convout
        return x_highway
    # END YOUR CODE


if __name__ == '__main__':
    eword = 10
    batch = 100
    highway = Highway(eword)
    x_convout = torch.randn(batch, eword)
    ret = highway.forward(x_convout)
    print(ret.shape)
