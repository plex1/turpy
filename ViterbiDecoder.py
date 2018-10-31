#! /usr/bin/env python
# title           : ViterbiDecoder.py
# description     : This class implements a viterbi decoder. Its input is a trellis instance.
# author          : Felix Arnold
# python_version  : 3.5.2


class ViterbiDecoder(object):

    def __init__(self, trellis):
        self.state = 0
        self.trellis = trellis
        self.terminated = True

    def decode(self, encoded_rx, n_data):

        trellis = self.trellis
        n_stages = int(n_data / self.trellis.wu)

        # forward state metric calculation
        sm_vec = [0] + [-10] * (trellis.Ns - 1)  # init state metric vector
        decisions = []
        for i in range(0, n_stages):  # for each stage
            decisions_stage = []
            sm_vec_new = []
            llr = encoded_rx[trellis.wc * i:trellis.wc * (i + 1)]
            for j in range(trellis.Ns):  # for each state
                branches = trellis.get_prev_branches_pc[j]
                sums = []
                for k in range(len(branches)):  # for each branch
                    branch_metric = 0
                    for l in range(trellis.wc):  # for each encoded bit
                        if trellis.get_enc_bits_pc[branches[k]][l] == 1:
                            branch_metric = branch_metric + llr[l]  # add
                    sums.append(sm_vec[trellis.get_prev_state_pc[branches[k]]] + branch_metric)
                decision = sums.index(max(sums))  # compare
                sm_vec_new.append(sums[decision])  # select
                decisions_stage.append(decision)
            sm_vec = list(sm_vec_new)
            decisions.append(decisions_stage)

        # traceback
        if self.terminated:
            state = 0  # start state when terminated trellis
        else:
            state = sm_vec.index(max(sm_vec))
        data_r = []
        for i in reversed(range(n_stages)):  # loop over all stages backwards
            decision = decisions[i][state]
            branch_taken = trellis.get_prev_branches_pc[state][decision]
            data_r = trellis.get_dat_pc[branch_taken] + data_r
            state = trellis.get_prev_state_pc[branch_taken]

        return data_r
