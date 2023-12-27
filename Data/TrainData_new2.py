from pathlib import Path
import socket
from tqdm import tqdm

import numpy as np
from skimage.io import imread
from skimage.color import rgb2ycbcr
from skimage.transform import rescale
import glob
import torch
from torch.utils.data import Dataset
# only use for test on BVI-DVC performance on a little high rate on Y loop filter
def yuv_import(filename, width, height, numfrm, startfrm=0):
    # Open the file
    f = open(filename, "rb")

    # Skip some frames
    luma_size = height * width
    chroma_size = luma_size // 4
    frame_size = luma_size * 3 // 2
    f.seek(frame_size * startfrm, 0)

    # Define the YUV buffer
    Y = np.zeros([numfrm, height, width], dtype=np.uint8)
    U = np.zeros([numfrm, height//2, width//2], dtype=np.uint8)
    V = np.zeros([numfrm, height//2, width//2], dtype=np.uint8)

    # Loop over the frames
    for i in range(numfrm):
        # Read the Y component
        Y[i, :, :] = np.fromfile(f, dtype=np.uint8, count=luma_size).reshape([height, width])
        # Read the U component
        U[i, :, :] = np.fromfile(f, dtype=np.uint8, count=chroma_size).reshape([height//2, width//2])
        # Read the V component
        V[i, :, :] = np.fromfile(f, dtype=np.uint8, count=chroma_size).reshape([height//2, width//2])

    # Close the file
    f.close()

    return Y, U, V


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
    #root = Path(r'/home/vcl/whduan/Datasets/images/')
    root = Path(r'/home/vcl/space/Datasets_yuv/images_yuv')
    #root = Path(r'/home/vcl/space/OpenImage_v6/train_00_yuv/')
    if not root.exists():
        raise ValueError("Path of dataset do not exist.")
    
    return root


# class TrainData(Dataset):
#     def __init__(self, block_size, debug=False):
#         super().__init__()
#         self._bs = block_size
#         self._images = list()

#         # Read the images
#         # root = get_addr(debug)

#         # files = [item for item in root.iterdir()]   
#         images_sum = list()
#         # dir_images = ['/home/vcl/space/OpenImage_v6/train_00_yuv/*.yuv','/home/vcl/space/OpenImage_v6/train_01_yuv/*.yuv',\
#         #     '/home/vcl/space/OpenImage_v6/train_02_yuv/*.yuv', '/home/vcl/space/OpenImage_v6/train_03_yuv/*.yuv','/home/vcl/space2/OpenImage_v6/train_05_yuv/*.yuv', '/home/vcl/space2/OpenImage_v6/train_06_yuv/*.yuv',\
#         #          '/home/vcl/space2/OpenImage_v6/train_07_yuv/*.yuv','/home/vcl/space2/OpenImage_v6/train_08_yuv/*.yuv']
#         dir_images = ['/backup2/whduan/vimeo_yuv_420p/*.yuv']
#         print(dir_images)
#         for i in dir_images:
#             image_list = glob.glob(i)
#             images_sum = images_sum + image_list
#         for file in tqdm(images_sum, desc="Load training dataset"):
#             # print(file)
#             self._images.append(file)
            
#             # image = imread(file)
#             # if image.shape[2] == 3:
#             #     self._images.append(image)

#     def __len__(self):
#         return len(self._images)

#     def __getitem__(self, index):
#         image = self._images[index]
#         w = 448
#         h = 256
#         Y, U, V = yuv_import(image, w, h, 1)
#         block_Y = np.squeeze(Y)/255.0
#         block_U = np.squeeze(U)/255.0
#         block_V = np.squeeze(V)/255.0

#         iw = 0
#         iw = np.random.randint(0, w//2-self._bs//2)*2
#         iw_down = iw // 2
#         # 在v2版本中为了训练，将vimeo数据集的大小patch size调整为256x256。
#         block_Y = block_Y[:, iw:iw+self._bs]
        
#         block_U = block_U[:, iw_down:iw_down+self._bs//2]
        
#         block_V = block_V[:, iw_down:iw_down+self._bs//2]
#         # random flip
#         if np.random.rand() < 0.5:
#             block_Y = block_Y[:, ::-1]
#             block_U = block_U[:, ::-1]
#             block_V = block_V[:, ::-1]
#         #2021.11.2移除了上下翻转
#         if np.random.rand() < 0.5:
#             block_Y = block_Y[::-1, :]
#             block_U = block_U[::-1, :]
#             block_V = block_V[::-1, :]
#         #convert to tensor
#         y_comp = torch.from_numpy(block_Y.astype(np.float32)).unsqueeze(0)
#         #print(y_comp.shape)
#         u_comp = torch.from_numpy(block_U.astype(np.float32)).unsqueeze(0)
#         v_comp = torch.from_numpy(block_V.astype(np.float32)).unsqueeze(0)
#         return y_comp, u_comp, v_comp
        

class TrainData2(Dataset):
    def __init__(self, block_size, debug=False):
        super().__init__()
        self._bs = block_size
        self._images = list()

        # Read the images
        # root = get_addr(debug)

        # files = [item for item in root.iterdir()]   
        images_sum = list()
        dir_images = ["/backup2/whduan/dataset_all/*.yuv"]
        print(dir_images)
        # 在BVI-DVC上fine-tune一下看看效果会不会变好
        for i in dir_images:
            image_list = glob.glob(i)
            images_sum = images_sum + image_list
        for file in tqdm(images_sum, desc="Load training dataset"):
            # print(file)
            self._images.append(file)
            
            # image = imread(file)
            # if image.shape[2] == 3:
            #     self._images.append(image)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):

        image = str(self._images[index])
        # name = image.split('/')[-1].split('.')[-2].split('_')[-1]
        name = image.split('/')[-1].split('.')[-2].split('_')[-2]
        w = int(name.split('x')[-2])
        h = int(name.split('x')[-1])

        image = self._images[index]
        Y, U, V = yuv_import(image, w, h, 1)

        ih = 0
        iw = 0
        if (h//2-self._bs//2) != 0:
            ih = np.random.randint(0, h//2-self._bs//2)*2
            iw = np.random.randint(0, w//2-self._bs//2)*2
        else:
            iw = np.random.randint(0, w//2-self._bs//2)*2

        ih_down = ih // 2
        iw_down = iw // 2
        #print(ih,ih_down)
        block_Y = np.squeeze(Y)/255.0
        block_U = np.squeeze(U)/255.0
        block_V = np.squeeze(V)/255.0

        block_Y = block_Y[ih:ih+self._bs, iw:iw+self._bs]
        
        block_U = block_U[ih_down:ih_down+self._bs//2, iw_down:iw_down+self._bs//2]
        
        block_V = block_V[ih_down:ih_down+self._bs//2, iw_down:iw_down+self._bs//2]
        
        # block_U = np.squeeze(U)[ih_down:ih_down+self._bs/2, iw_down:iw_down+self._bs/2]
        # block_V = np.squeeze(V)[ih_down:ih_down+self._bs/2, iw_down:iw_down+self._bs/2]

        #random flip 加入了水平和垂直翻转
        if np.random.rand() < 0.5:
            block_Y = block_Y[:, ::-1]
            block_U = block_U[:, ::-1]
            block_V = block_V[:, ::-1]

        # if np.random.rand() < 0.5:
        #     block_Y = block_Y[::-1, :]
        #     block_U = block_U[::-1, :]
        #     block_V = block_V[::-1, :]
        
        #convert to tensor
        y_comp = torch.from_numpy(block_Y.astype(np.float32)).unsqueeze(0)
        #print(y_comp.shape)
        u_comp = torch.from_numpy(block_U.astype(np.float32)).unsqueeze(0)
        v_comp = torch.from_numpy(block_V.astype(np.float32)).unsqueeze(0)
        return y_comp, u_comp, v_comp