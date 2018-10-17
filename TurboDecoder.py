#! /usr/bin/env python
# title           : TurboDecoder.py
# description     : This class implements a turbo decoder. Its input is an interleaver and two siso decoder instances.
# author          : Felix Arnold
# python_version  : 3.5.2

import numpy as np


class TurboDecoder(object):

    def __init__(self, interleaver, convsiso_p1, convsiso_p2):

        self.convsiso_p1 = convsiso_p1
        self.convsiso_p2 = convsiso_p2
        self.il = interleaver
        self.n_zp = 3  # zero padding
        self.iterations = 6

    def decode(self, ys, yp1, yp2, data_u=[]):

        n_data = len(ys) - self.n_zp
        # initialize variables
        Lext = [0] * n_data
        ext_scale = 0.75
        ys = list(np.array(ys))
        ys_i = ys[0:-self.n_zp]  # systematic bits through interleaver
        zp = [-10] * self.n_zp  # zero padding (the trellis is not terminated but zero padded)
        il = self.il

        errors_iter = [0] * self.iterations

        for i in range(self.iterations):

            # first half iteration
            data_r_sisop1 = self.convsiso_p1.decode(ys, yp1, il.deinterleave(Lext) + zp, n_data)
            Lext = list(ext_scale * (np.array(data_r_sisop1) - np.array(il.deinterleave(Lext)) - np.array(ys_i)))

            # second half iteration
            data_r_sisop2 = self.convsiso_p2.decode(il.interleave(ys_i) + zp, yp2, il.interleave(Lext) + zp, n_data)
            Lext = list(ext_scale * (
                    np.array(data_r_sisop2) - np.array(il.interleave(Lext)) - il.interleave(np.array(ys[0:-3]))))

            # hard output
            data_r_sisop2_th = il.deinterleave((np.array(data_r_sisop2) > 0).astype(int))  # threshold

            if len(data_u) > 0:  # ber calcuation
                data_r_sisop1_th = (np.array(data_r_sisop1) > 0).astype(int)  # threshold
                errors_p1 = (np.array(data_u) != data_r_sisop1_th).sum()
                errors_p2 = (np.array(data_u) != data_r_sisop2_th).sum()
                errors_iter[i] = errors_p2
                if errors_p2 == 0:  # stopping criteria
                    return (data_r_sisop2_th, errors_iter)

        return (data_r_sisop2_th, errors_iter)
