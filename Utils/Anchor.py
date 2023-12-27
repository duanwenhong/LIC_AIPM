import os

import h5py
import numpy as np
from matplotlib import pyplot as plt


class Anchor(object):
    def __init__(self, threshold=3.5):
        super().__init__()
        self.threshold = threshold

        self.data = dict()
        root = os.path.split(os.path.abspath(__file__))[0]
        with h5py.File(os.path.join(root, "Results.h5"), 'r') as f:
            for item in f.keys():
                data = f[item][:]
                self.data[item] = data[:, np.argsort(data[0, :])]

    def plot(self, bpp=None, psnr=None):
        fig = plt.figure(1)
        for key in self.data:
            rates = self.data[key][0, :]
            psnrs = self.data[key][1, :]
            index = rates < self.threshold
            rates = rates[index]
            psnrs = psnrs[index]
            plt.plot(rates, psnrs, label=key)
        plt.title("RD Performance", fontsize=14)
        plt.xlabel("BPP", fontsize=12)
        plt.ylabel("PSNR (dB)", fontsize=12)
        if bpp is not None and psnr is not None:
            plt.plot(bpp, psnr, "rx")
        if isinstance(bpp, list):
            for i in range(len(bpp)):
                plt.plot(bpp[i], psnr[i], 'x')
        else:
            plt.plot(bpp, psnr)
        plt.grid(True)
        plt.legend()
        
        return fig


if __name__ == "__main__":
    anchor = Anchor(1)
    anchor.plot(0.4, 36)
    plt.show()
