import argparse
import sys
import os
import scipy
import torch
import cv2
from utils import *
from src import models
from src.model import *
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
sys.path.append(os.getcwd())

def generate_density_map(mat_file_path, scale = 4, kernel_size = 16, sigma = 3):
    mat = scipy.io.loadmat(mat_file_path)
    
    # 提取点标注数据 (x, y)
    points = mat['image_info'][0, 0][0, 0][0]
    h, w = 768, 1024

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

def main(args):
    density = generate_density_map(args.mat_file_path)
    
    plt.imshow(density, cmap='jet', interpolation='nearest')

    # 添加颜色条（color bar）
    plt.colorbar()

    # 设置标题
    plt.title("Density Map Heatmap (Blue to Red)")

    # 保存为 PNG 文件
    plt.savefig('density_map_heatmap_.png', dpi=300, bbox_inches='tight')

    # 关闭图形，释放内存
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_file_path',type=str, default= "imgs/ShanghaiTech/part_A/train_data/ground-truth/GT_IMG_15.mat")
    parser.add_argument('--model_cof',type=str, default="cof/eticn.yml")
    parser.add_argument('--output_path',type=str, default="ret.png")
    args = parser.parse_args()
    main(args)