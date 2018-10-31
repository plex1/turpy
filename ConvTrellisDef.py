#! /usr/bin/env python
# title           : ConvTrellis.py
# description     : This class implements functions on a trellis of a convolutional code.
#                   Its input is the generator polynomal of the code.
# author          : Felix Arnold
# python_version  : 3.5.2


import numpy as np
from utils import dec2bin
from utils import get_bit


class ConvTrellisDef(object):

    def __init__(self, gen_matrix_norm, gen_feedback_norm=[]):

        self.gen_matrix = np.fliplr(np.array(gen_matrix_norm))
        self.K = self.get_k()  # constraint length
        self.Ns = 2 ** (self.K - 1)  # number of states
        self.Nb = 2 ** self.K  # number of branches
        self.wc = self.get_rate()
        self.wu = 1  # 1 data bit per stage
        self.gen_feedback = np.fliplr(np.array([gen_feedback_norm]))
        self.rsc = len(gen_feedback_norm) > 0

    def get_rate(self):
        return self.gen_matrix.shape[0]

    def get_k(self):
        return self.gen_matrix.shape[1]

    def get_next_state(self, branch):
        return branch >> 1

    def get_prev_state(self, branch):
        return branch & (2 ** (self.K - 1) - 1)

    def get_prev_branches(self, state):
        return np.array([state << 1, (state << 1) + 1])

    def get_next_branches(self, state):
        return np.array([state, 2 ** (self.K - 1) + state])

    def get_enc_bits(self, branch):
        return list(np.mod(np.matmul(self.gen_matrix, (dec2bin(branch, self.K))), 2))

    def get_dat(self, branch):
        if not self.rsc:
            return [int((branch & (2 ** (self.K - 1))) > 0)]
        else:
            return list(
                np.mod(np.matmul(self.gen_feedback, (dec2bin(branch, self.K))) + get_bit(branch, self.K - 1), 2))
