import numpy as np


class Trellis(object):

    def __init__(self, gen_matrix):
        if not isinstance(gen_matrix, np.ndarray):
            print('generator matrix has to be of type numpy.array')
            return
        else:
            self.gen_matrix = gen_matrix
            self.K = self.get_k()  # constraint length
            self.Ns = 2 ** (self.K - 1)  # number of states
            self.Nb = 2 ** self.K  # number of branches
            self.r = self.get_rate()

    def get_rate(self):
        return self.gen_matrix.shape[0]

    def get_k(self):
        return self.gen_matrix.shape[1]

    def get_next_state(self, branch):
        return branch & (2**(self.K-1)-1)

    def get_prev_state(self, branch):
        return branch >> 1

    def get_prev_branches(self, state):
        return np.array([state, 2 ** (self.K - 1) + state])

    def get_prev_branch(self, state, dat):
        return self.get_prev_branches(state)[dat]

    def get_next_branches(self, state):
        return np.array([state << 1, (state << 1) + 1])

    def get_next_branch(self, state, dat):
        return self.get_next_branches(state)[dat]

    def get_enc_bits(self, branch):
        return np.mod(np.matmul(self.gen_matrix, (self.dec2bin(branch, self.K))), 2)

    def get_prev_dat(self, state):
        return state & 1

    def dec2bin(self, val, k):
        bin_val = []
        for j in range(k):
            bin_val.append(val & 1)
            val = val >> 1
        return bin_val
