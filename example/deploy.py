import argparse
import base64
from fastapi import FastAPI, File, UploadFile
import torch
import numpy as np
from fastapi.responses import JSONResponse
import cv2
import uvicorn

from src.utils import *
from src import *

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Crowd Density Estimation System"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        binary_data = await file.read()
        nparr = np.frombuffer(binary_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  
        if image is None:
            return JSONResponse(content={"error": "Uploaded file is not a valid image."}, status_code=400)
        density_map = model.inference(image_data = image)
        _, encoded_image = cv2.imencode(".png", density_map)
        encoded_image = base64.b64encode(encoded_image).decode("utf-8")
        return JSONResponse(
            content={"message": "Image processed successfully.","image":encoded_image},
            headers={"Content-Type": "application/octet-stream"}
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def main(args):
    global model
    config = load_config(args.model_cof)
    model = models[config["model_type"]](**config["model"])
    model.load_pretrained(config["logging"]["save_dir"])  
    uvicorn.run(app, host = args.host, port = args.port, reload=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_cof',type = str, default = "config/density.yml")
    parser.add_argument('--host', type = str, default = "127.0.0.1")
    parser.add_argument('--port', type = int, default = 8000)
    args = parser.parse_args()
    main(args)
    