#! /usr/bin/env python
# title           : ConvTrellis.py
# description     : This class implements functions on a trellis of a convolutional code.
#                   Its input is the generator polynomal of the code.
# author          : Felix Arnold
# python_version  : 3.5.2


import numpy as np
from utils import dec2bin


class ConvTrellisDef(object):

    def __init__(self, gen_matrix, gen_feedback=[]):

        self.gen_matrix = np.array(gen_matrix)
        self.K = self.get_k()  # constraint length
        self.Ns = 2 ** (self.K - 1)  # number of states
        self.Nb = 2 ** self.K  # number of branches
        self.wc = self.get_rate()
        self.wu = 1  # 1 data bit per stage
        self.r = self.get_rate()
        self.gen_feedback = np.array(gen_feedback)
        self.rsc = len(gen_feedback) > 0

    def get_rate(self):
        return self.gen_matrix.shape[0]

    def get_k(self):
        return self.gen_matrix.shape[1]

    def get_next_state(self, branch):
        return branch & (2 ** (self.K - 1) - 1)

    def get_prev_state(self, branch):
        return branch >> 1

    def get_prev_branches(self, state):
        return np.array([state, 2 ** (self.K - 1) + state])

    def get_prev_branch(self, state, dat):  # obsolete
        return self.get_prev_branches(state)[dat]

    def get_next_branches(self, state):
        return np.array([state << 1, (state << 1) + 1])

    def get_next_branch(self, state, dat):
        if not self.rsc:
            return self.get_next_branches(state)[dat]
        else:
            return (state << 1) + np.mod(np.matmul(self.gen_feedback, (dec2bin(state << 1, self.K))) + dat, 2)

    def get_enc_bits(self, branch):
        return list(np.mod(np.matmul(self.gen_matrix, (dec2bin(branch, self.K))), 2))

    def get_prev_dat(self, state):  # obsolete
        return state & 1

    def get_dat(self, branch):
        if not self.rsc:
            return [branch & 1]
        else:
            return [np.mod(np.matmul(self.gen_feedback, (dec2bin(branch, self.K))) + (branch & 1), 2)]
