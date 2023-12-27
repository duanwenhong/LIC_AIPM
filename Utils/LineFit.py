from pathlib import Path

import h5py
import numpy as np


class LineFit(object):
    def __init__(self):
        super().__init__()
        self.data = dict()
        with h5py.File(Path(__file__).parent / "Results.h5", 'r') as f:
            for item in f.keys():
                data = f[item][:]
                self.data[item] = data[:, np.argsort(data[0, :])]

    def __call__(self, value, key):
        rates = self.data[key][0, :]
        psnrs = self.data[key][1, :]
        upper_index = np.searchsorted(rates, value, "left").clip(min=1, max=len(rates)-1)

        low_rate = rates[upper_index-1]
        up_rate = rates[upper_index]
        low_psnr = psnrs[upper_index-1]
        up_psnr = psnrs[upper_index]

        interpolate = (up_psnr-low_psnr) / (up_rate-low_rate) * (value-low_rate) + low_psnr

        return interpolate


if __name__ == "__main__":
    line_fit = LineFit()
    value = np.random.rand() * 3
    print("Rate    : {:>.4f} bpp;".format(value))
    print("JPEG    : {:>5.2f} dB;".format(line_fit(value, "JPEG")))
    print("OPENJPEG: {:>5.2f} dB;".format(line_fit(value, "OPENJPEG")))
    print("JPEG2K  : {:>5.2f} dB;".format(line_fit(value, "JPEG2K")))
    print("HM420   : {:>5.2f} dB;".format(line_fit(value, "HM420")))
    print("HM444   : {:>5.2f} dB;".format(line_fit(value, "HM444")))
    print("VTM420  : {:>5.2f} dB;".format(line_fit(value, "VTM420")))
