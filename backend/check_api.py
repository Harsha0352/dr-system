import requests
import os

url = "http://127.0.0.1:8000/predict"
image_path = r"d:\DRDS_HARSHA\dr_unified_v2\dr_unified_v2\train\0\10000_right.jpg"

if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    exit(1)

try:
    print(f"Sending request to {url}...")
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Error Response: {response.text}")

except Exception as e:
    print(f"Exception: {e}")
