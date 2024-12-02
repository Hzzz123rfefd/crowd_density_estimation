import argparse
import sys
import os
import torch
import cv2
from utils import *
from src import models
from src.model import *
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())



def main(args):
    """ get model """
    config = load_config(args.model_cof)
    net = models[config["model_type"]](**config["model"])
    net.load_pretrained(config["logging"]["save_dir"])  
    
    # load image
    image = cv2.imread(args.image_path)
    image = torch.tensor(image).permute(2, 0, 1).float()

    density = net.get_density(image)
    density = density.squeeze(0).numpy()
    plt.imshow(density, cmap='jet', interpolation='nearest')

    # 添加颜色条（color bar）
    plt.colorbar()

    # 设置标题
    plt.title("Density Map Heatmap (Blue to Red)")

    # 保存为 PNG 文件
    plt.savefig('density_map_heatmap.png', dpi=300, bbox_inches='tight')

    # 关闭图形，释放内存
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',type=str, default= "imgs/ShanghaiTech/part_B/train_data/images/IMG_15.jpg")
    parser.add_argument('--model_cof',type=str, default="cof/eticn.yml")
    parser.add_argument('--output_path',type=str, default="ret.png")
    args = parser.parse_args()
    main(args)