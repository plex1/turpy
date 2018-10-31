from ConvTrellisDef import ConvTrellisDef
from Trellis import Trellis
from ConvEncoder import ConvEncoder
from ViterbiDecoder import ViterbiDecoder
from ConvEncoder import TurboEncoder
from SisoDecoder import SisoDecoder
from Interleaver import Interleaver
from TurboDecoder import TurboDecoder

import numpy as np


def test_scripts():
    # test if the scripts run without error
    # the output of the scripts is not checked
    import ViterbiTest
    ViterbiTest.main(20, False, False)

    import TurboTest
    block_size = 32
    n_blocks = 1
    TurboTest.main(block_size, n_blocks, False, False)


def test_turbo():
    d = [1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]

    # parameters
    gp_forward = [[1, 1, 0, 1]]
    gp_feedback = [0, 0, 1, 1]

    n_zp = 6

    trellis_identity = Trellis(ConvTrellisDef([[1]]))
    trellis_p = Trellis(ConvTrellisDef(gp_forward, gp_feedback))

    # create interleaver instances
    il = Interleaver()
    il.gen_rand_perm(len(d))

    # create encoder and decoder instances
    trellises = [trellis_identity, trellis_p, trellis_p]
    turboenc = TurboEncoder(trellises, il)
    turboenc.n_zp = n_zp
    csiso = SisoDecoder(trellis_p)
    csiso.backward_init = False
    td = TurboDecoder(il, csiso, csiso)
    td.n_zp = n_zp

    # encode
    encoded_streams = turboenc.encode(d)

    # check encoder
    # systematic bits
    zp = [0] * n_zp
    assert d + zp == encoded_streams[0]

    # parity bits 1
    convenc = ConvEncoder(trellis_p)
    e = convenc.encode(d + zp, False)
    assert e == encoded_streams[1]

    # parity bits 2
    e = convenc.encode(il.interleave(d) + zp, False)
    assert e == encoded_streams[2]

    # introduce errors
    e = np.array(turboenc.flatten(encoded_streams))
    for i in range(0, len(e), 10):
        e[i] = int(not (e[i]))

    # to llr calculation
    e = [2 * x - 1 for x in e]

    # check decoder
    [ys, yp1, yp2] = turboenc.extract(e)  # extract streams
    drx, errors = td.decode(ys, yp1, yp2, d)

    assert d == drx
    assert errors[-1] == 0


def test_interleaver():
    N = 40
    k1 = 3
    k2 = 10
    il = Interleaver()

    # QPP interleaver

    il.gen_qpp_perm_poly(N, k1, k2)

    # check that deinterleave(interleave(x)) == x
    d = list(range(N))
    di = il.interleave(d)
    did = il.deinterleave(di)

    assert d[0] == di[0]
    assert d[-1] != di[-1]
    assert d == did

    # random interleaver

    il.gen_rand_perm(N)
    di = il.interleave(d)
    did = il.deinterleave(di)
    assert d == did


def test_siso():
    # encode
    g = [[1, 0, 0], [1, 1, 1]]
    d = [1, 0, 1, 0, 0, 1]

    trellis = Trellis(ConvTrellisDef(g))
    convenc = ConvEncoder(trellis)

    e = convenc.encode(d, True)

    # flip bits
    e[1] = int(not (e[1]))
    e[-1] = int(not (e[-1]))

    # decode with siso decoder
    convsiso = SisoDecoder(trellis)
    n_stages = len(d) + trellis.tdef.K - 1
    data_r, c = convsiso.decode([0] * n_stages, e, n_stages)

    # compare outputs to reference
    minf = convsiso.minus_inf
    assert [2, -2, 2, -2, -2, 2, -1 + minf, minf] == data_r
    assert [2, 2, -2, 2, 2, -2, -2, 2, -2, 2, 2, 2, -1 + minf, 2, minf, 2] == c

    # make threshold and check correct decoding of message
    data_r = convenc.remove_zero_termination(data_r)
    data_r = list((np.array(data_r) > 0).astype(int))  # threshold
    assert d == data_r

    # reduction 2 = radix 4
    trellisr2 = Trellis(ConvTrellisDef(g), 2)
    convsiso = SisoDecoder(trellisr2)

    data_r, c = convsiso.decode([0] * n_stages, e, n_stages)

    # compare outputs to reference
    minf = convsiso.minus_inf
    assert [2, -2, 2, -2, -2, 2, -1 + minf, minf] == data_r
    assert [2, 2, -2, 2, 2, -2, -2, 2, -2, 2, 2, 2, -1 + minf, 2, minf, 2] == c


def test_viterbi():
    # no feedback
    g = [[1, 0, 0], [1, 1, 1]]
    d = [1, 0, 1, 0, 0, 1]

    trellis = Trellis(ConvTrellisDef(g))
    convenc = ConvEncoder(trellis)

    e = convenc.encode(d, True)

    # flip bits
    e[1] = int(not (e[1]))
    e[-1] = int(not (e[-1]))

    # decode
    viterbi = ViterbiDecoder(trellis)
    data_r = viterbi.decode(e, len(d) + trellis.tdef.K - 1)
    data_r = convenc.remove_zero_termination(data_r)
    assert d == data_r

    # decode with radix 16
    reduction = 4
    trellis = Trellis(ConvTrellisDef(g), reduction)
    viterbi = ViterbiDecoder(trellis)
    data_r = viterbi.decode(e, len(d) + trellis.tdef.K - 1)
    data_r = convenc.remove_zero_termination(data_r)
    assert d == data_r

    # rate 1/3
    g = [[1, 0, 0], [1, 1, 1], [1, 0, 1]]
    trellis = Trellis(ConvTrellisDef(g))
    convenc = ConvEncoder(trellis)

    e = convenc.encode(d, True)

    # flip bits
    e[1] = int(not (e[1]))
    e[-1] = int(not (e[-1]))
    e[2] = int(not (e[2]))

    # decode
    viterbi = ViterbiDecoder(trellis)
    data_r = viterbi.decode(e, len(d) + trellis.tdef.K - 1)
    data_r = convenc.remove_zero_termination(data_r)
    assert d == data_r

    # with feedback, without termination
    g = [[1, 0, 0], [1, 1, 1]]
    fb = [0, 0, 1]

    trellis = Trellis(ConvTrellisDef(g, fb))
    convenc = ConvEncoder(trellis)

    # no termination
    e = convenc.encode(d, False)

    # flip bits
    e[1] = int(not (e[1]))
    e[-1] = int(not (e[-1]))

    # decode (viterbi decoder, not terminated)
    viterbi = ViterbiDecoder(trellis)
    viterbi.terminated = False
    data_r = viterbi.decode(e, len(d))
    assert d == data_r


def test_conv_enc():
    g = [[1, 0, 0], [1, 1, 1]]
    d = [1, 0, 1, 1, 0, 0]

    # no feedback
    trellis = Trellis(ConvTrellisDef(g))
    convenc = ConvEncoder(trellis)
    e = convenc.encode(d, False)
    assert [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1] == e

    # with feedback
    fb = [0, 0, 1]
    trellis = Trellis(ConvTrellisDef(g, fb))
    convenc = ConvEncoder(trellis)
    e = convenc.encode(d, False)
    assert [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0] == e

    # no feedback, radix 4 (reduction = 2)
    reduction = 2
    trellis = Trellis(ConvTrellisDef(g), reduction)
    convenc = ConvEncoder(trellis)
    e = convenc.encode(d, False)
    assert [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1] == e

    # with feedback , radix 4 (reduction = 2)
    fb = [0, 0, 1]
    trellis = Trellis(ConvTrellisDef(g, fb), reduction)
    convenc = ConvEncoder(trellis)
    e = convenc.encode(d, False)
    assert [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0] == e


def test_conftrellisdef():
    g = [[1, 0, 1], [1, 1, 0]]
    ctd = ConvTrellisDef(g)
    rate = ctd.get_rate()
    assert 2 == rate

    # non recursive code
    g = [[1, 0, 0], [1, 1, 1]]
    ctd = ConvTrellisDef(g)

    # test next branches
    b = ctd.get_next_branches(0)
    assert 2 == len(b)
    assert 0 == b[0]
    assert 1 == b[1]

    b = ctd.get_next_branches(1)
    assert 2 == len(b)
    assert 2 == b[0]
    assert 3 == b[1]

    # test next state
    s = ctd.get_next_state(4)
    assert 0 == s

    s = ctd.get_next_state(2)
    assert 2 == s

    # encoded bits
    e = ctd.get_enc_bits(3)
    assert [1, 0] == e

    e = ctd.get_enc_bits(4)
    assert [0, 1] == e

    # data bits
    d = ctd.get_dat(1)
    assert [1] == d

    d = ctd.get_dat(6)
    assert [0] == d

    # -----------------------------
    # recursive code
    g = [[1, 0, 0], [1, 1, 1]]
    fb = [0, 0, 1]
    ctd = ConvTrellisDef(g, fb)

    # data bits
    d = ctd.get_dat(6)
    assert [1] == d

    d = ctd.get_dat(4)
    assert [1] == d

    d = ctd.get_dat(2)
    assert [0] == d

    d = ctd.get_dat(7)
    assert [0] == d


def test_trellis():
    g = [[1, 0, 1], [1, 1, 0]]
    ct = Trellis(ConvTrellisDef(g))
    rate = ct.get_rate()
    assert 2 == rate

    # non recursive code
    g = [[1, 0, 0], [1, 1, 1]]
    ct = Trellis(ConvTrellisDef(g))

    # test next branches
    b = ct.get_next_branches_pc[0]
    assert 2 == len(b)
    assert 0 == b[0]
    assert 1 == b[1]

    b = ct.get_next_branches_pc[1]
    assert 2 == len(b)
    assert 2 == b[0]
    assert 3 == b[1]

    # test next state
    s = ct.get_next_state_pc[4]
    assert 0 == s

    s = ct.get_next_state_pc[2]
    assert 2 == s

    # encoded bits
    e = ct.get_enc_bits_pc[3]
    assert [1, 0] == e

    e = ct.get_enc_bits_pc[4]
    assert [0, 1] == e

    # data bits
    d = ct.get_dat_pc[1]
    assert [1] == d

    d = ct.get_dat_pc[6]
    assert [0] == d

    # -----------------------------
    # recursive code
    g = [[1, 0, 0], [1, 1, 1]]
    fb = [0, 0, 1]
    ct = Trellis(ConvTrellisDef(g, fb))

    # data bits
    d = ct.get_dat_pc[6]
    assert [1] == d

    d = ct.get_dat_pc[4]
    assert [1] == d

    d = ct.get_dat_pc[2]
    assert [0] == d

    d = ct.get_dat_pc[7]
    assert [0] == d

    s = 2
    b = ct.get_next_branches_pc[s]
    assert 4 == b[1]
