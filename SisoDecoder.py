#! /usr/bin/env python
# title           : SisoDecoder.py
# description     : This class implements soft input soft output decoder for a convolutional code.
#                   Its input is a trellis instance. The max-log-BCJR algorithm is employed.
# author          : Felix Arnold
# python_version  : 3.5.2


class SisoDecoder(object):

    def __init__(self, trellis):
        self.state = 0
        self.trellis = trellis
        self.remove_tail = True
        self.forward_init = True
        self.backward_init = False

    def decode(self, input_u, input_c, n_data):

        minf = -10
        trellis = self.trellis
        n_stages = int(n_data / self.trellis.wb)
        sm_vec_init = [0] + [minf * self.forward_init] * (trellis.Ns - 1)  # init state metric vector

        # forward (alpha)
        sm_vec = sm_vec_init
        sm_forward = []
        for i in range(0, n_stages):  # for each stage
            sm_vec_new = []
            llr = input_c[trellis.r * i:trellis.r * (i + 1)]
            ysp = input_u[trellis.wb * i: trellis.wb * (i + 1)]
            for j in range(trellis.Ns):  # for each state
                branches = trellis.get_prev_branches_pc[j]
                branch_sums = []
                for k in range(len(branches)):  # for each branch
                    branch_metric = 0
                    for l in range(trellis.r):  # for each encoded bit
                        if trellis.get_enc_bits_pc[branches[k]][l] == 1:
                            branch_metric += llr[l]
                    for l in range(trellis.wb):  # for each data bit
                        if trellis.get_dat_pc[branches[k]][l]:
                            branch_metric += ysp[l]
                    branch_sums.append(sm_vec[trellis.get_prev_state_pc[branches[k]]] + branch_metric)  # add (gamma)
                sm_vec_new.append(max(branch_sums))  # compare and select
            sm_vec = list(sm_vec_new)
            sm_forward.append(sm_vec)

        # backward (beta)
        sm_backward = []
        output_u = []
        output_c = []
        sm_vec = [0] + [minf * self.backward_init] * (trellis.Ns - 1)  # init state metric vector
        for i in reversed(range(0, n_stages)):  # for each stage
            sm_vec_new = []
            llr = input_c[trellis.r * i:trellis.r * (i + 1)]
            ysp = input_u[trellis.wb * i: trellis.wb * (i + 1)]
            max_branch = [[minf, minf] for i in range(trellis.wb)]
            max_branch_enc = [[minf, minf] for i in range(trellis.r)]
            for j in range(trellis.Ns):  # for each state
                branches = trellis.get_next_branches_pc[j]
                branch_sums = []
                for k in range(len(branches)):  # for each branch
                    branch_metric = 0
                    for l in range(trellis.r):  # for each encoded bit
                        if trellis.get_enc_bits_pc[branches[k]][l] == 1:
                            branch_metric += llr[l]
                    for l in range(trellis.wb):  # for each data bit
                        if trellis.get_dat_pc[branches[k]][l]:
                            branch_metric += ysp[l]
                    branch_sum = sm_vec[trellis.get_next_state_pc[branches[k]]] + branch_metric # add (gamma)
                    branch_sums.append(branch_sum)

                    # add the state metric from the forward propagation -> total = alpha + gamma + beta
                    if i == 0:
                        total_metric = branch_sum + sm_vec_init[j]
                    else:
                        total_metric = branch_sum + sm_forward[i - 1][j]

                    # soft encoded output calculation
                    enc = trellis.get_enc_bits_pc[branches[k]]
                    for n in range(trellis.r):
                        if total_metric > max_branch_enc[n][enc[n]]:
                            max_branch_enc[n][enc[n]] = total_metric

                    # soft data output calculation
                    out = trellis.get_dat_pc[branches[k]]
                    for n in range(trellis.wb):
                        if total_metric > max_branch[n][out[n]]:
                            max_branch[n][out[n]] = total_metric

                sm_vec_new.append(max(branch_sums))  # compare and select

            sm_vec = list(sm_vec_new)
            sm_backward.insert(0, sm_vec)
            if i < n_data or not self.remove_tail:  # soft output
                for n in reversed(range(trellis.wb)):
                    output_u.insert(0, max_branch[n][1] - max_branch[n][0])

            for n in reversed(range(trellis.r)): # soft encoded output
                output_c.insert(0, max_branch_enc[n][1] - max_branch_enc[n][0])

        return (output_u, output_c)
