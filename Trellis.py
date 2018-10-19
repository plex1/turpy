#! /usr/bin/env python
# title           : Trellis.py
# description     : This class implements functions on a trellis of a convolutional code.
#                   Its input is the generator polynomal of the code.
# author          : Felix Arnold
# python_version  : 3.5.2

import numpy as np


class Trellis(object):

    def __init__(self, gen_matrix, gen_feedback=[]):
        self.gen_matrix = np.array(gen_matrix)
        self.K = self.get_k()  # constraint length
        self.Ns = 2 ** (self.K - 1)  # number of states
        self.Nb = 2 ** self.K  # number of branches
        self.r = self.get_rate()
        self.gen_feedback = np.array(gen_feedback)
        self.rsc = len(gen_feedback) > 0
        self.get_dat_precalc = []
        self.get_enc_bits_precalc = []
        self.optimize = False
        self.precalc()

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
            return (state << 1) + np.mod(np.matmul(self.gen_feedback, (self._dec2bin(state << 1, self.K))) + dat, 2)

    def get_enc_bits(self, branch):
        if not self.optimize:
            return np.mod(np.matmul(self.gen_matrix, (self._dec2bin(branch, self.K))), 2)
        else:
            return self.get_enc_bits_pc[branch]

    def get_prev_dat(self, state):  # obsolete
        return state & 1

    def get_dat(self, branch):
        if not self.optimize:
            if not self.rsc:
                return branch & 1
            else:
                return np.mod(np.matmul(self.gen_feedback, (self._dec2bin(branch, self.K))) + (branch & 1), 2)
        else:
            return self.get_dat_pc[branch]

    def _dec2bin(self, val, k):
        bin_val = []
        for j in range(k):
            bin_val.append(val & 1)
            val = val >> 1
        return bin_val

    def precalc(self):
        self.get_dat_pc=[]
        self.get_enc_bits_pc = []
        for i in range(self.Nb):
            self.get_dat_pc.append(self.get_dat(i))
            self.get_enc_bits_pc.append(self.get_enc_bits(i))

        self.get_dat_pc = [self.get_dat(x) for x in range(self.Nb)]
        self.get_enc_bits_pc = [self.get_enc_bits(x) for x in range(self.Nb)]
        self.get_next_state_pc = [self.get_next_state(x) for x in range(self.Nb)]
        self.get_prev_state_pc = [self.get_prev_state(x) for x in range(self.Nb)]
        self.get_next_branches_pc = [self.get_next_branches(x) for x in range(self.Nb)]
        self.get_prev_branches_pc = [self.get_prev_branches(x) for x in range(self.Nb)]

        self.optimize = True
