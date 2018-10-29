#! /usr/bin/env python
# title           : Interleaver.py
# description     : This class implements an interleaver.
# author          : Felix Arnold
# python_version  : 3.5.2

import random
import math


class Interleaver(object):

    def __init__(self):
        self.perm = []
        self.perm_inv = []

    def get_length(self):
        return len(self.perm)

    def set_permutation(self, perm):
        self.perm = perm

    def gen_no_perm(self, l):
        self.perm = list(range(l))
        self._gen_perm_inv()

    def gen_rev_perm(self, l):
        self.perm = list(reversed(range(l)))
        self._gen_perm_inv()

    def gen_rand_perm(self, l):
        self.perm = list(range(l))
        random.shuffle(self.perm)
        self._gen_perm_inv()

    def gen_qpp_perm(self, N):
        # possible values for N = power of two
        self.perm = list(range(N))
        k = int((math.log2(N) + 1) / 2)
        self.perm = [((2 ** k - 1) * x + 2 ** (k + 1) * x ** 2) % N for x in self.perm]
        self._gen_perm_inv()

    def gen_qpp_perm_poly(self, N, k1, k2):
        self.perm = list(range(N))
        self.perm = [(k1 * x + k2 * x ** 2) % N for x in self.perm]
        self._gen_perm_inv()

    def _gen_perm_inv(self):
        self.perm_inv = [0] * len(self.perm)
        for i in range(len(self.perm)):
            self.perm_inv[self.perm[i]] = i

    def interleave(self, data):
        return [data[index] for index in self.perm]

    def deinterleave(self, data):
        return [data[index] for index in self.perm_inv]
