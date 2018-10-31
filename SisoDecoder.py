#! /usr/bin/env python
# title           : SisoDecoder.py
# description     : This class implements soft input soft output decoder for specified trellis.
#                   Its input is a trellis instance. The max-log-BCJR algorithm is employed.
# author          : Felix Arnold
# python_version  : 3.5.2


class SisoDecoder(object):

    def __init__(self, trellis):
        self.state = 0
        self.trellis = trellis
        self.forward_init = True
        self.backward_init = True
        self.minus_inf = -10

    def decode(self, input_u, input_c, n_data):

        minus_inf = self.minus_inf
        trellis = self.trellis
        n_stages = int(n_data / self.trellis.wu)
        sm_vec_init = [0] + [minus_inf * self.forward_init] * (trellis.Ns - 1)  # init state metric vector

        # forward (alpha)
        sm_vec = sm_vec_init
        sm_forward = []
        for i in range(0, n_stages):  # for each stage
            sm_vec_new = []
            cin = input_c[trellis.wc * i:trellis.wc * (i + 1)]
            uin = input_u[trellis.wu * i: trellis.wu * (i + 1)]
            for j in range(trellis.Ns):  # for each state
                branches = trellis.get_prev_branches_pc[j]
                branch_sums = []
                for k in range(len(branches)):  # for each branch
                    branch_metric = 0
                    for l in range(trellis.wc):  # for each encoded bit
                        if trellis.get_enc_bits_pc[branches[k]][l] == 1:
                            branch_metric += cin[l]
                    for l in range(trellis.wu):  # for each data bit
                        if trellis.get_dat_pc[branches[k]][l]:
                            branch_metric += uin[l]
                    branch_sums.append(sm_vec[trellis.get_prev_state_pc[branches[k]]] + branch_metric)  # add (gamma)
                sm_vec_new.append(max(branch_sums))  # compare and select
            sm_vec = list(sm_vec_new)
            sm_forward.append(sm_vec)

        # backward (beta)
        output_u = []
        output_c = []
        sm_vec = [0] + [minus_inf * self.backward_init] * (trellis.Ns - 1)  # init state metric vector

        for i in reversed(range(0, n_stages)):  # for each stage
            sm_vec_new = []
            cin = input_c[trellis.wc * i:trellis.wc * (i + 1)]
            uin = input_u[trellis.wu * i: trellis.wu * (i + 1)]
            max_branch_dat = [[minus_inf, minus_inf] for i in range(trellis.wu)]
            max_branch_enc = [[minus_inf, minus_inf] for i in range(trellis.wc)]
            for j in range(trellis.Ns):  # for each state
                branches = trellis.get_next_branches_pc[j]
                branch_sums = []
                for k in range(len(branches)):  # for each branch
                    branch_metric = 0
                    for l in range(trellis.wc):  # for each encoded bit
                        if trellis.get_enc_bits_pc[branches[k]][l] == 1:
                            branch_metric += cin[l]
                    for l in range(trellis.wu):  # for each data bit
                        if trellis.get_dat_pc[branches[k]][l]:
                            branch_metric += uin[l]
                    branch_sum = sm_vec[trellis.get_next_state_pc[branches[k]]] + branch_metric  # add (gamma)
                    branch_sums.append(branch_sum)

                    # add the state metric from the forward propagation -> total = alpha + gamma + beta
                    if i == 0:
                        total_metric = branch_sum + sm_vec_init[j]
                    else:
                        total_metric = branch_sum + sm_forward[i - 1][j]

                    # soft encoded output calculation
                    enc = trellis.get_enc_bits_pc[branches[k]]
                    for n in range(trellis.wc):
                        if total_metric > max_branch_enc[n][enc[n]]:
                            max_branch_enc[n][enc[n]] = total_metric

                    # soft data output calculation
                    dat = trellis.get_dat_pc[branches[k]]
                    for n in range(trellis.wu):
                        if total_metric > max_branch_dat[n][dat[n]]:
                            max_branch_dat[n][dat[n]] = total_metric

                sm_vec_new.append(max(branch_sums))  # compare and select

            sm_vec = list(sm_vec_new)

            for n in reversed(range(trellis.wu)):  # soft output
                output_u.insert(0, max_branch_dat[n][1] - max_branch_dat[n][0])

            for n in reversed(range(trellis.wc)):  # soft encoded output
                output_c.insert(0, max_branch_enc[n][1] - max_branch_enc[n][0])

        return output_u, output_c
