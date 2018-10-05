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

    def encode(self, data):
        # add K-1 zero termination
        data = np.concatenate((data, np.array([0] * (self.trellis.K - 1))))

        # encoding
        encoded = np.array([])
        for i in range(len(data)):
            encoded = np.concatenate((encoded.astype(int), self.step(data[i])))

        return encoded
