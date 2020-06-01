#!/usr/bin/env python

import os
import fnmatch
import numpy as np
from matplotlib import pyplot as plt


def main():

    fname = None
    dir = './test/test1'
    for file in os.listdir(dir):
        if fnmatch.fnmatch(file, '*.txt'):
            fname = dir+'/'+file
    assert fname is not None

    y_true, y_pred = np.loadtxt(fname, int, unpack=True)

    fig = plt.figure()
    plt.plot(y_true, label='ytrue')
    plt.plot(y_pred*0.9, label='ypred')
    plt.xlim([0,1000])
    plt.ylim([-0.2,1.2])
    plt.title('ytrue vs ypred')
    plt.legend()
    plt.draw()
    plt.savefig(fname.replace('.txt','png'))

if __name__ == '__main__':
    main()
