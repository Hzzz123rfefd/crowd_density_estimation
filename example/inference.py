import argparse
import sys
import os
import cv2

from src.utils import *
from src import models
from src.model import *
sys.path.append(os.getcwd())



def main(args):
    """ get model """
    config = load_config(args.model_cof)
    net = models[config["model_type"]](**config["model"])
    net.load_pretrained(config["logging"]["save_dir"])  
    
    """ inference """
    density = net.inference(image_path = args.image_path)
    cv2.imwrite(args.output_path, density)
    # density_map = cv2.resize(density_map, (image.shape[1], image.shape[0]))
    # density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
    # density_map = cv2.applyColorMap(density_map.astype(np.uint8), cv2.COLORMAP_JET)
    # density_map = density_map.astype(np.uint8)


    # plt.figure(figsize=(8, 6))
    # plt.hist(flattened_density, bins=200, color='blue', edgecolor='black', alpha=0.7)
    # plt.xlabel('Density Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Density Values')
    # plt.show()

    # plt.imshow(density, cmap='jet', interpolation='nearest')
    # plt.colorbar()
    # plt.title("Density Map Heatmap ")
    # plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
    # plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',type=str, default= "imgs/ShanghaiTech/part_B/train_data/images/IMG_15.jpg")
    parser.add_argument('--model_cof',type=str, default="config/density.yml")
    parser.add_argument('--output_path',type=str, default="ret.png")
    args = parser.parse_args()
    main(args)