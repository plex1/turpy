import numpy as np
from numpy.random import rand, randn
from Trellis import Trellis
from ConvEncoder import ConvEncoder
from ViterbiDecoder import  ViterbiDecoder
import matplotlib.pyplot as plt

# parameters
n_data = 20000
EbNodB_range = np.arange(-6, 6, 2)
code_choice = 1
gp_list = {  # definition of generator polynomals
    1: np.array([[1, 0, 1], [1, 1, 1]]),  # constraint length K=3, Rate=1/2
    2: np.array([[1, 0, 1],[1, 1, 1],[1, 1, 0]]),  # constraint length K=3, Rate=1/3
    3: np.array([[1, 0, 0, 1, 1, 1, 1],[1, 1, 0, 1, 1, 0, 1]])  # constraint length K=7, Rate=1/2
    }

# create instances
gen_poly = gp_list[code_choice]
trellis = Trellis(gen_poly)
convenc = ConvEncoder(trellis)
viterbi = ViterbiDecoder(trellis)

# loop over all SNRs
ber_vec = []
ber_vec_raw = []
for n in range(0, len(EbNodB_range)):

    print("-------- EbN0: " + str(EbNodB_range[n]) + " -------------")

    # generate data
    print("Generate User Data..")
    data_u = (rand(n_data) >= 0.5).astype(int)

    # convolutional encoding
    encoded = convenc.encode(data_u)

    # additive noise
    EbNo = 10.0 ** (EbNodB_range[n]/ 10.0)
    noise_std = np.sqrt(trellis.r) / np.sqrt(2 * EbNo)
    encoded_rx = 2 * encoded - 1 + noise_std * randn(len(encoded))

    # viterbi decoding
    print("Viterbi Decoding..")
    data_r=viterbi.decode(encoded_rx, n_data)

    # ber calculation
    print("BER calculation..")
    ber_vec.append((data_u != data_r).sum() / len(data_r))
    encoded_rxth = (encoded_rx >= 0)  # threshold for calculation of un-coded ber
    ber_vec_raw.append((encoded != encoded_rxth).sum() / len(encoded))
    print("ber_raw=" + str(ber_vec_raw[-1]))
    print("ber_coded=" + str(ber_vec[-1]))
    print("bit errors coded: " + str((data_u != data_r).sum()))

# summary output
print("BER Vector: " + str(ber_vec))
EbNodB_range_raw = EbNodB_range - 10 * np.log10(trellis.r)  # the raw (un-coded) transmission has another EbNo
plt.plot(EbNodB_range_raw, ber_vec_raw, '-r')
plt.plot(EbNodB_range, ber_vec, '-b')
plt.plot(EbNodB_range_raw, ber_vec_raw, 'ro')
plt.plot(EbNodB_range, ber_vec, 'bo')
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('EbNo(dB)')
plt.ylabel('BER')
plt.grid(True)
plt.title('BPSK modulation, Convolutional Code, Soft Viterbi Decoding')
plt.legend(('Uncoded', 'Coded'))
plt.show()

