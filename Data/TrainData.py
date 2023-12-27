from pathlib import Path
import socket
from tqdm import tqdm

import numpy as np
from skimage.io import imread
from skimage.color import rgb2ycbcr
from skimage.transform import rescale

import torch
from torch.utils.data import Dataset

def get_addr(debug):
    # Addrs = {
    #     "162.105.23.213": r"G:\Datasets\images(png)",
    #     "162.105.94.226": r"/Extended/backup1/zhzhao/Datasets/images(png)",
    #     "162.105.94.223": r"E:\Datasets\images(png)",
    #     "162.105.94.93": r"/backup/home/zhzhao/Datasets/images(png)",
    #     "162.105.94.50": r"/backup2/zhzhao/Datasets/images(png)",
    # }

    # s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # try:
    #     s.connect(('8.8.8.8', 80))
    #     ip = s.getsockname()[0]
    # finally:
    #     s.close()
    # root = Addrs[ip] if not debug else Addrs[ip]+"-debug"
    # root = Path(root)

    #duan modify train dataset
    root = Path(r'/home/vcl/whduan/Datasets/images/')
    if not root.exists():
        raise ValueError("Path of dataset do not exist.")
    
    return root


class TrainData(Dataset):
    def __init__(self, block_size, debug=False):
        super().__init__()
        self._bs = block_size
        self._images = list()

        # Read the images
        root = get_addr(debug)
        files = [item for item in root.iterdir()]
        for file in tqdm(files, desc="Load training dataset"):
            image = imread(file)
            if image.shape[2] == 3:
                self._images.append(image)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = self._images[index]
        h, w, c = image.shape

        # Crop the image
        ih = np.random.randint(0, h-self._bs)
        iw = np.random.randint(0, w-self._bs)
        block = image[ih:ih+self._bs, iw:iw+self._bs, :]

        # Random flip the image
        if np.random.rand() < 0.5:
            block = block[:, ::-1, :]

        # Convert to yuv420p
        yuv = rgb2ycbcr(block) / 255.0
        y_comp = yuv[:, :, 0]
        u_comp = rescale(yuv[:, :, 1], 0.5, anti_aliasing=True)
        v_comp = rescale(yuv[:, :, 2], 0.5, anti_aliasing=True)

        # Convert to Tensor
        y_comp = torch.from_numpy(y_comp).unsqueeze(0)
        u_comp = torch.from_numpy(u_comp).unsqueeze(0)
        v_comp = torch.from_numpy(v_comp).unsqueeze(0)

        return y_comp, u_comp, v_comp
