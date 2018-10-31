#! /usr/bin/env python
# title           : TurboTest.py
# description     : This script tests the turbo decoding for parallel concatenated convolutional codes
# author          : Felix Arnold
# python_version  : 3.5.2

import numpy as np
from numpy.random import rand, randn
from scipy.stats import norm
import matplotlib.pyplot as plt
from Trellis import Trellis
from ConvTrellisDef import ConvTrellisDef
from ConvEncoder import TurboEncoder
from SisoDecoder import SisoDecoder
from Interleaver import Interleaver
from TurboDecoder import TurboDecoder


def main(n_data=512, n_blocks=10, verbose=True, do_plot=True):
    # parameters
    EbNodB_range = [-1, 0, 0.8, 1, 1.2, 1.3]
    gp_forward = [[1, 1, 0, 1]]
    gp_feedback = [0, 0, 1, 1]

    # create interleaver instances
    il = Interleaver()
    il.gen_qpp_perm(n_data)

    # create trellises, encoders and decoders instances
    trellis_p = Trellis(ConvTrellisDef(gp_forward, gp_feedback))
    trellis_identity = Trellis(ConvTrellisDef([[1]]))
    csiso = SisoDecoder(trellis_p)
    csiso.backward_init = False
    trellises = [trellis_identity, trellis_p, trellis_p]
    turboenc = TurboEncoder(trellises, il)
    td = TurboDecoder(il, csiso, csiso)

    # loop over all SNRs
    error_vec = []
    blocks_vec = []
    for EbNodB in EbNodB_range:

        if verbose:
            print("----- Simulating EbN0 = " + str(EbNodB) + " -----")

        # loop over several code blocks
        errors_acc = [0] * td.iterations
        blocks = 0
        for k in range(n_blocks):

            blocks += 1

            # generate data
            data_u = list((rand(n_data) >= 0.5).astype(int))

            # turbo encoding
            encoded_streams = turboenc.encode(data_u)
            encoded = np.array(turboenc.flatten(encoded_streams))

            # additive noise
            EbNo = 10.0 ** (EbNodB / 10.0)
            noise_std = np.sqrt(2 + 1) / np.sqrt(2 * EbNo)
            encoded_rx = 2 * encoded - 1 + noise_std * randn(len(encoded))

            # turbo decoding
            [ys, yp1, yp2] = turboenc.extract(encoded_rx)  # extract streams
            d, errors = td.decode(ys, yp1, yp2, data_u)
            if verbose:
                print('block               : ' + str(k))
                print('errors per iteration: ' + str(errors))
            errors_acc = list(np.array(errors_acc) + np.array(errors))

            if errors_acc[3] > 3 * n_data:  # simulation stopping criteria
                break

        error_vec.append(errors_acc)
        blocks_vec.append(blocks)

    # summary output and plot
    error_vec_t = list(np.transpose(np.array(error_vec)))
    if verbose:
        print("Simulated errors: " + str(error_vec_t))
    if do_plot:
        for i in range(0, td.iterations):
            ber = list(np.array(error_vec_t[i]) / (np.array(blocks_vec) * n_data))
            plt.plot(EbNodB_range, ber, 'x-')
        ber_uncoded = norm.sf(np.sqrt(2 * np.array(10 ** (np.array(EbNodB_range) / 10))))
        plt.plot(EbNodB_range, ber_uncoded, 'k:')
        plt.xscale('linear')
        plt.yscale('log', nonposy='mask')
        plt.xlabel('EbNo(dB)')
        plt.ylabel('BER')
        plt.grid(True, which="both")
        plt.title('BPSK modulation, Turbo Code, max log MAP decoding')
        plt.show()


if __name__ == "__main__":
    main()
