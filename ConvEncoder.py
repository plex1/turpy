#! /usr/bin/env python
# title           : ConvEncoder.py
# description     : This class implements a convolutional encoder. Its input is a trellis instance.
# author          : Felix Arnold
# python_version  : 3.5.2

import numpy as np


class ConvEncoder(object):

    def __init__(self, trellis):
        self.state = 0
        self.trellis = trellis

    def reset(self):
        self.state = 0

    def step(self, data):
        branch_taken = self.trellis.get_next_branch(self.state, data)
        out = self.trellis.get_enc_bits(branch_taken)
        self.state = self.trellis.get_next_state(branch_taken)
        return out

    def get_state(self):
        return self.state

    def zero_padding(self, data, k=-1):
        # add k-1 zero termination bits
        if k == -1:
            k = self.trellis.K - 1
        return data + [0] * k

    def encode(self, data, zero_termination=True):

        self.reset()

        if zero_termination:
            data = self.zero_padding(data)

        # encoding
        encoded = np.array([])
        for i in range(len(data)):
            encoded = np.concatenate((encoded.astype(int), self.step(data[i])))

        return encoded


class TurboEncoder(object):

    def __init__(self, trellises, interleaver):
        self.state = 0
        self.trellises = trellises
        self.interleaver = interleaver
        self.r = 3

    def encode(self, data, K=0):

        self.cvet = ConvEncoder(self.trellises[1])  # trellis with maximum K

        encoded = []
        for index, trellis in enumerate(self.trellises):
            self.cve = ConvEncoder(trellis)
            self.cve.reset()
            datam = data
            if index > 1:
                datam = self.interleaver.interleave(datam)
            datam = self.cvet.zero_padding(datam, K)
            encoded_conv = self.cve.encode(datam, False)
            encoded.append(encoded_conv)
        return encoded

    def extract(self, enc_stream):
        enc_extracted = []
        for i in range(self.r):
            enc_extracted.append(enc_stream[i::self.r])
        return enc_extracted

    def flatten(self, enc_extracted):
        enc_stream = []
        for i in range(len(enc_extracted[0])):
            for j in range(self.r):
                enc_stream.append(enc_extracted[j][i])
        return enc_stream