import pytest
from fastapi.testclient import TestClient
import sys
import os
import io                     
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app

client = TestClient(app)

def test_predict_valid_image():
    # สร้างรูปภาพจำลองสีแดงขนาด 224x224
    img = Image.new('RGB', (224, 224), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    response = client.post(
        "/predict", 
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction_class" in data
    assert "confidence" in data

def test_predict_invalid_file_type():
    # จำลองการส่งไฟล์ Text
    response = client.post(
        "/predict", 
        files={"file": ("test.txt", b"Hello World, this is not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

def test_predict_corrupted_image():
    # จำลองการส่งไฟล์ที่นามสกุลเป็นรูป แต่ไส้ในไม่ใช่รูป
    response = client.post(
        "/predict", 
        files={"file": ("test.jpg", b"fake image bytes", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "Corrupted" in response.json()["detail"]
    
def test_predict_large_file():
    # จำลองไฟล์ขนาด 6MB (เกินลิมิต 5MB)
    large_bytes = b"0" * (6 * 1024 * 1024)
    response = client.post(
        "/predict", 
        files={"file": ("large.jpg", large_bytes, "image/jpeg")}
    )
    assert response.status_code == 413 # 413 Request Entity Too Large