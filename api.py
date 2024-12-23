import base64
import json
import requests
import time

start_time = time.time()
api_url = "http://127.0.0.1:8000/predict/"

with open("a/output_0001.png", "rb") as img_file:
    files = {"file": img_file}
    response = requests.post(api_url, files=files)

if response.status_code == 200:
    response_str = response.content.decode("utf-8")
    response_data = json.loads(response_str)
    encoded_image = response_data["image"]
    image_data = base64.b64decode(encoded_image)
    with open("density_map.png", "wb") as output_file:
        output_file.write(image_data)
    print("Density map saved as 'density_map.png'")
else:
    print(f"Error: {response.status_code} - {response.text}")
sum_result = sum(range(1, 1000000))

# 程序结束时间
end_time = time.time()

# 计算耗时（单位：秒）
elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒

print(f"程序运行时间: {elapsed_time:.2f} 毫秒")