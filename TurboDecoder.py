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

    def decode(self, ys, yp1, yp2, expected_data=[]):

        # initialize variables
        n_data = len(ys)
        Lext = [0] * n_data
        ext_scale = 11/16
        ys_i = ys[0:-self.n_zp]  # systematic bits without zero padding (interleaved bits)
        zp = [-10] * self.n_zp  # zero padding (the trellis is not terminated to zero but zero padded)
        il = self.il

        errors_iter = [0] * self.iterations

        for i in range(self.iterations):

            # first half iteration ------------------------------------------------

            # prepare apriori information
            input_u = [x + y for x, y in zip(ys, il.deinterleave(Lext)+zp)]

            # decode
            dec1, cout = self.convsiso_p1.decode(input_u , yp1, n_data)

            #  calculate extrinsic information
            Lext = list(ext_scale * (np.array(dec1[0:-self.n_zp]) - np.array(il.deinterleave(Lext)) - np.array(ys_i)))

            # second half iteration ------------------------------------------------

            # prepare apriori information
            input_u =  [x + y for x, y in zip(il.interleave(ys_i) + zp, il.interleave(Lext)+zp)]

            # decode
            dec2, cout = self.convsiso_p2.decode(input_u, yp2, n_data)

            #  calculate extrinsic information
            Lext = list(ext_scale * (np.array(dec2[0:-self.n_zp]) - np.array(il.interleave(Lext)) - il.interleave(np.array(ys_i))))

            # hard output
            dec_out = il.deinterleave((np.array(dec2) > 0).astype(int))  # threshold

            if len(expected_data) > 0:  # ber calculation
                errors = (np.array(expected_data) != dec_out).sum()
                errors_iter[i] = errors
                if errors == 0:  # stopping criteria
                    return (dec_out, errors_iter)

        return (dec_out, errors_iter)
