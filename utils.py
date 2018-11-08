#! /usr/bin/env python
# title           : utils.py
# description     : This is a file for small utility functions used in this package
# author          : Felix Arnold
# python_version  : 3.5.2

from itertools import islice


def dec2bin(val, k):
    bin_val = []
    for j in range(k):
        bin_val.append(val & 1)
        val = val >> 1
    return bin_val


def bin2dec(bin_val):
    n = len(bin_val)
    int_val = 0
    for j in range(n):
        int_val += bin_val[j] * 2 ** j
    return int_val


def get_bit(v, n):
    return int((v & (2 ** n)) > 0)


def get_bits(v, start, end):
    mask = 2 ** (end - start + 1) - 1
    return (v >> start) & mask


def grouped(seq, n):
    it = iter(seq)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            # StopIteration
            return
        yield chunk


flatten = lambda l: (item for sublist in l for item in sublist)
