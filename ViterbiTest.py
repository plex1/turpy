#! /usr/bin/env python
# title           : ViterbiTest.py
# description     : This script performs convolutional encoding and viterbi decoding using BPSK on an AWGN channel
# author          : Felix Arnold
# python_version  : 3.5.2

import numpy as np
from numpy.random import rand, randn
from scipy.stats import norm
from Trellis import Trellis
from ConvEncoder import ConvEncoder
from ViterbiDecoder import ViterbiDecoder
import matplotlib.pyplot as plt
from ConvSISO import ConvSISO


def main(n_data=5000, verbose=True):
    # parameters
    EbNodB_range = np.arange(-4, 10, 2)
    code_choice = 2
    gp_list = {  # definition of generator polynomals
        1: [[1, 0, 1], [1, 1, 1]],  # constraint length K=3, Rate=1/2
        2: [[1, 0, 1], [1, 1, 1], [1, 1, 0]],  # constraint length K=3, Rate=1/3
        3: [[1, 0, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 0, 1]]  # constraint length K=7, Rate=1/2
    }
    gen_feedback = []
    use_viterbi = True

    # create instances
    gen_poly = gp_list[code_choice]
    trellis = Trellis(gen_poly, gen_feedback)
    convenc = ConvEncoder(trellis)
    viterbi = ViterbiDecoder(trellis)

    # loop over all SNRs
    ber_vec = []
    ber_vec_raw = []
    for n in range(0, len(EbNodB_range)):

        if verbose:
            print("Simulating EbN0: " + str(EbNodB_range[n]))

        # generate data
        data_u = list((rand(n_data) >= 0.5).astype(int))

        # convolutional encoding
        encoded = convenc.encode(data_u)

        # additive noise
        EbNo = 10.0 ** (EbNodB_range[n] / 10.0)
        noise_std = np.sqrt(trellis.r) / np.sqrt(2 * EbNo)
        encoded_rx = 2 * np.array(encoded) - 1 + noise_std * randn(len(encoded))

        # viterbi decoding
        if use_viterbi:
            data_r = viterbi.decode(encoded_rx, n_data)
        else:  # siso
            convsiso = ConvSISO(trellis)
            convsiso.backward_init = True
            n_stages = n_data + trellis.K - 1
            data_r, c = convsiso.decode([0] * n_stages, encoded_rx, n_data)
            data_r = (np.array(data_r) > 0).astype(int)

        # ber calculation
        ber_vec.append((np.array(data_u) != data_r).sum() / len(data_r))
        encoded_rxth = (encoded_rx >= 0)  # threshold for calculation of un-coded ber
        ber_vec_raw.append((encoded != encoded_rxth).sum() / len(encoded))

    # summary output
    if verbose:
        print("Simulated BER values: " + str(ber_vec))
    EbNodB_range_raw = EbNodB_range - 10 * np.log10(trellis.r)  # the raw (un-coded) transmission has another EbNo
    plt.plot(EbNodB_range_raw, ber_vec_raw, '-r', marker='x')
    plt.plot(EbNodB_range, ber_vec, '-b', marker='o')
    ber_uncoded = norm.sf(np.sqrt(2 * np.array(10 ** (EbNodB_range / 10))))
    plt.plot(EbNodB_range, ber_uncoded, ':r')
    plt.xscale('linear')
    plt.yscale('log', nonposy='mask')
    plt.xlabel('EbNo(dB)')
    plt.ylabel('BER')
    plt.grid(True, which="both")
    plt.title('BPSK modulation, Convolutional Code, Soft Viterbi Decoding')
    plt.legend(('Uncoded', 'Coded'))
    plt.show()


if __name__ == "__main__":
    main()
