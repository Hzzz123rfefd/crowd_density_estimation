import json
import streamlit as st
import requests
import os
import time
import cv2
import numpy as np
import base64
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_column_width.*")

API_URL = "http://127.0.0.1:8000/predict/"

image_folder = "./stream"  
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')])

st.title("实时人群密度检测模拟系统v1.0")


start_button = st.button("开始模拟")

if start_button:
    video_placeholder = st.empty()

    for i, frame_file in enumerate(image_files):
        frame_path = os.path.join(image_folder, frame_file)
        start_time = time.time()
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)  

        with open(frame_path, "rb") as f:
            files = {"file": f}
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                response_str = response.content.decode("utf-8")
                response_data = json.loads(response_str)
                encoded_image = response_data["image"]
                image_data = base64.b64decode(encoded_image)

                # 将解码后的数据转为 OpenCV 图像
                np_img = np.frombuffer(image_data, dtype=np.uint8)
                heatmap_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)  # 热力图图像

                # 将热力图转换为 RGB 格式（OpenCV 默认 BGR）
                heatmap_img_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

                # 显示实时视频流和热力图
                video_placeholder.image(heatmap_img_rgb, channels="RGB", caption=f"模拟监控视频流", use_container_width=True)
            else:
                st.error("Error processing the frame.")

            frame_time = time.time() - start_time
            sleep_time = max(0, (1 / 15) - frame_time)
            time.sleep(sleep_time)

    st.write("结束模拟")
