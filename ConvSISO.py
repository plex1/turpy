#! /usr/bin/env python
# title           : ConvSISO.py
# description     : This class implements soft input soft output decoder for a convolutional code.
#                   Its input is a trellis instance. The max-log-BCJR algorithm is employed.
# author          : Felix Arnold
# python_version  : 3.5.2


class ConvSISO(object):

    def __init__(self, trellis):
        self.state = 0
        self.trellis = trellis
        self.remove_tail = True
        self.forward_init = True
        self.backward_init = False

    def decode(self, ys, yp, la, n_data, ):

        trellis = self.trellis
        n_stages = n_data + trellis.K - 1
        sm_vec_init = [0] + [-10 * self.forward_init] * (trellis.Ns - 1)  # init state metric vector

        # forward (alpha)
        sm_vec = sm_vec_init
        sm_forward = []
        for i in range(0, n_stages):  # for each stage
            sm_vec_new = []
            llr = yp[trellis.r * i:trellis.r * (i + 1)]
            for j in range(trellis.Ns):  # for each state
                branches = trellis.get_prev_branches_pc[j]
                sums = []
                for k in range(2):  # for each branch
                    branch_metric = 0
                    for l in range(trellis.r):  # for each encoded bit
                        if trellis.get_enc_bits_pc[branches[k]][l] == 1:
                            branch_metric += llr[l]
                    if trellis.get_dat_pc[branches[k]]:
                        branch_metric += ys[i] + la[i]
                    sums.append(sm_vec[trellis.get_prev_state_pc[branches[k]]] + branch_metric)  # add
                decision = int(sums[1] > sums[0])  # compare
                sm_vec_new.append(sums[decision])  # select
            sm_vec = list(sm_vec_new)
            sm_forward.append(sm_vec)

        # backward (beta)
        sm_backward = []
        lu = []
        lue = []
        sm_vec = [0] + [-10 * self.backward_init] * (
                trellis.Ns - 1)  # init state metric vector # init state metric vector
        for i in reversed(range(0, n_stages)):  # for each stage
            decisions_stage = []
            sm_vec_new = []
            llr = yp[trellis.r * i:trellis.r * (i + 1)]
            max_branch = [-10, -10]
            max_branch_enc = [[-10, -10] for i in range(trellis.r)]
            for j in range(trellis.Ns):  # for each state
                branches = trellis.get_next_branches_pc[j]
                sums = []
                for k in range(2):  # for each branch
                    branch_metric = 0
                    for l in range(trellis.r):  # for each encoded bit
                        if trellis.get_enc_bits_pc[branches[k]][l] == 1:
                            branch_metric += llr[l]
                    if trellis.get_dat_pc[branches[k]]:
                        branch_metric += ys[i] + la[i]
                    branch_sum = sm_vec[trellis.get_next_state_pc[branches[k]]] + branch_metric  # add (gamma)
                    sums.append(branch_sum)

                    if i == 0:
                        post_branch_metric = branch_sum + sm_vec_init[j]
                    else:
                        post_branch_metric = branch_sum + sm_forward[i - 1][j]

                    # soft encoded output calculation
                    enc = trellis.get_enc_bits_pc[branches[k]]
                    for n in range(trellis.r):
                        if post_branch_metric > max_branch_enc[n][enc[n]]:
                            max_branch_enc[n][enc[n]] = post_branch_metric

                    # soft data output calculation
                    out = trellis.get_dat_pc[branches[k]]
                    if post_branch_metric > max_branch[out]:
                        max_branch[out] = post_branch_metric

                decision = int(sums[1] > sums[0])  # compare
                sm_vec_new.append(sums[decision])  # select
                decisions_stage.append(decision)

            sm_vec = list(sm_vec_new)
            sm_backward.insert(0, sm_vec)
            if i < n_data or not self.remove_tail:  # soft output
                lu.append(max_branch[1] - max_branch[0])

            for n in reversed(range(trellis.r)): # soft encoded output
                lue.append(max_branch_enc[n][1] - max_branch_enc[n][0])

        lu = list(reversed(lu))
        self.lue = list(reversed(lue))

        return lu
