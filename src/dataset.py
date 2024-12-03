import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import cv2

from src.utils import *

class DatasetForImageReader(Dataset):
    def __init__(
        self, 
        train_image_folder:str = None,
        test_image_folder:str = None,
        valid_image_folder:str = None,
        data_type:str = "train"
    ):
        self.data_type = data_type
        if data_type == "train":
            self.use_image_folder = train_image_folder
        elif data_type == "test":
            self.use_image_folder = test_image_folder
        elif data_type == "valid":
            self.use_image_folder = valid_image_folder

        image_data = []
        files = os.listdir(self.use_image_folder)
        files = [f for f in files if os.path.isfile(os.path.join(self.use_image_folder, f))]
        files = sorted(files, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

        for filename in files:
            img_file_path = os.path.join(self.use_image_folder, filename)
            image_data.append(cv2.imread(img_file_path))
        
        a = np.array(image_data)
        self.dataset = np.transpose(np.array(image_data), (0, 3, 1, 2))
        self.dataset = self.dataset / 255.0
        self.image_height = self.dataset.shape[2]
        self.image_weight = self.dataset.shape[3]
        self.data_type = data_type
        self.total_samples = len(self.dataset)

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        output = {}
        output["images"] = torch.tensor(self.dataset[idx], dtype=torch.float32)
        return output
    
    def collate_fn(self,batch):
        return recursive_collate_fn(batch)
    
class DatasetForDensity(DatasetForImageReader):
    def __init__(
            self, 
            train_image_folder = None,
            test_image_folder = None,
            valid_image_folder = None,
            train_label_folder = None, 
            test_label_folder = None, 
            valid_label_folder = None,
            data_type = "train"
    ):
        super().__init__(train_image_folder, test_image_folder, valid_image_folder, data_type)
        if self.data_type == "train":
            self.use_label_folder = train_label_folder
        elif self.data_type == "test":
            self.use_label_folder = test_label_folder
        elif self.data_type == "valid":
            self.use_label_folder = valid_label_folder
        
        label_data = []
        files = os.listdir(self.use_label_folder)
        files = [f for f in files if os.path.isfile(os.path.join(self.use_label_folder, f))]
        files = sorted(files, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
        for filename in files:
            mat_file_path = os.path.join(self.use_label_folder, filename)
            label_data.append(self.generate_density_map(mat_file_path))

        self.label = np.array(label_data)


    def generate_density_map(self,mat_file_path, scale = 4, kernel_size = 16, sigma = 3):
        mat = scipy.io.loadmat(mat_file_path)
        
        # 提取点标注数据 (x, y)
        points = mat['image_info'][0, 0][0, 0][0]
        h, w = self.image_height, self.image_weight

        # 初始化空白密度图
        density_map = np.zeros((h, w), dtype=np.float32)

        # 提取
        for point in points:
            x, y = int(point[0]), int(point[1])
            
            if 0 <= x < w and 0 <= y < h:
                density_map[y, x] += 1

        assert h % scale == 0 and w % scale == 0, "Height and width should be divisible by" + str(scale)
        scale_density_map = density_map.reshape(h // scale, scale, w // scale, scale).sum(axis=(1, 3))
        kernel = gaussian_kernel(kernel_size, sigma)
        gaussian_density_map = convolve(scale_density_map, kernel, mode='reflect')

        return gaussian_density_map
    
    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        output = {}
        output["image"] = torch.tensor(self.dataset[idx], dtype=torch.float32)
        output["label"] = torch.tensor(self.label[idx], dtype=torch.float32)
        return output