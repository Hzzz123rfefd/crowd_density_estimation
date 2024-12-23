import base64
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import numpy as np
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from io import BytesIO
from PIL import Image

from utils import *
from src import *

def inference(model, image, target_width, target_height):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, (target_width, target_height))
    resized_image = torch.tensor(image).permute(2, 0, 1).float()
    density_map = model.get_density(resized_image)
    density_map = density_map.squeeze(0).numpy()
    density_map = cv2.resize(density_map, (image.shape[1], image.shape[0]))
    density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
    density_map = density_map.astype(np.uint8)
    return density_map

config = load_config("cof/eticn.yml")
model = models[config["model_type"]](**config["model"])
model.load_pretrained(config["logging"]["save_dir"])  
print("model load success!")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Crowd Density Estimation API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        binary_data = await file.read()
        nparr = np.frombuffer(binary_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  
        print(image.shape)
        print(image)
        if image is None:
            return JSONResponse(content={"error": "Uploaded file is not a valid image."}, status_code=400)
        print("get image")
        density_map = inference(model, image, 1024, 768)
        print(density_map.shape)
        print(density_map)
        _, encoded_image = cv2.imencode(".png", density_map)
        encoded_image = base64.b64encode(encoded_image).decode("utf-8")
        print("density_map")
        return JSONResponse(
            content={"message": "Image processed successfully.","image":encoded_image},
            headers={"Content-Type": "application/octet-stream"}
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
