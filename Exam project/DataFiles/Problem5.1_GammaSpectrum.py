#!/usr/bin/env python

from array import array
import numpy as np



# ----------------------------------------------------------------------------------- #
# Read data (channel numbers):
# ----------------------------------------------------------------------------------- #

x = []

with open( 'data_GammaSpectrum.txt', 'r' ) as infile :
    for line in infile:
        line = line.strip().split()
        x.append(float(line[0]))

        # Print the numbers as a sanity check
        if (len(x) < 10) :
            print(x[-1])

x = np.array(x)