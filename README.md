---
title: Dog Cat Classification API
emoji: 🐶
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
---

# 🚀 High-Throughput Dog vs Cat Classification API

โปรเจกต์นี้เป็นระบบบริการจำแนกรูปภาพสุนัขและแมว (Image Classification) ที่ออกแบบมาเพื่อรองรับการใช้งานพร้อมกันจำนวนมาก (High-Throughput) โดยใช้เทคนิค MLOps ครบวงจร ตั้งแต่การทำ Model Optimization ไปจนถึงการทำ CI/CD และ Containerization

## 📂 โครงสร้างโปรเจกต์ (Project Structure)
```text
project/
├── app/
│   └── main.py          # ระบบ API ด้วย FastAPI รองรับ Concurrency (Worker Pool)
├── model/
│   ├── hf_resnet50_prep.onnx   # โมเดลหลักที่ใช้งาน (Graph Optimized) 🚀
│   ├── hf_resnet50_int8.onnx   # โมเดล Quantized
│   ├── hf_resnet50.onnx        # โมเดล ONNX ต้นฉบับ
│   ├── hf_resnet50.pt          # โมเดล PyTorch ต้นฉบับ
│   └── model_optimization.ipynb # บันทึกการทดลองและ Benchmark ผลลัพธ์
├── dataset/             # ชุดข้อมูลตัวอย่างสำหรับทดสอบ (Dog/Cat)
│   ├── cat/
│   └── dog/
├── test/
│   └── test_app.py      # ระบบ Unit Test (Pytest)
├── Dockerfile           # พิมพ์เขียวสำหรับสร้างระบบจำลอง (Container)
└── requirements.txt     # รายชื่อไลบรารีที่จำเป็น
```

## 🚀 การติดตั้งและใช้งาน (Getting Started)

### 1. การใช้งานบนเครื่อง Local
```bash
# ติดตั้งไลบรารี
pip install -r requirements.txt

# เริ่มต้นเซิร์ฟเวอร์
uvicorn app.main:app --reload
```
*เข้าใช้งาน Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)*

### 2. การรัน Unit Test
```bash
pytest test/test_app.py
```

### 3. การใช้งานผ่าน Docker (Containerization)
```bash
# สร้าง Image
docker build -t dog-cat-api .

# เปิดใช้งาน (พอร์ต 8000)
docker run -p 8000:8000 dog-cat-api
```

## ☁️ การเรียกใช้งาน API บน Cloud (Hugging Face Spaces)

คุณสามารถเรียกใช้งาน API ที่ออนไลน์อยู่ได้ทันทีผ่านคำสั่ง cURL ดังนี้:

```bash
curl -X 'POST' \
  'https://katanyapat-dog-cat-api.hf.space/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@dataset/cat/cat_1.jpg;type=image/jpeg'
```

### ตัวอย่าง JSON Response
```JSON
{
  "prediction_class": "Cat",
  "confidence": 0.8772
}
```

## 📊 ผลการทดสอบประสิทธิภาพ (Benchmarking)
| Model Type | File Size | Avg Latency (ms) |
| :--- | :--- | :--- |
| PyTorch (Baseline) | ~98 MB | ~30-40 ms |
| **ONNX Optimized (Prep)** | **~98 MB** | **~12.9 ms** |
| ONNX Quantized (INT8) | ~24 MB | ~18.9 ms |

## 🤖 ระบบอัตโนมัติ (CI/CD Pipeline)
ระบบรองรับการทำงานอัตโนมัติผ่าน GitHub Actions:
1. **Continuous Integration (CI):** รัน Unit Test อัตโนมัติทุกครั้งที่มีการ Push โค้ด
2. **Continuous Deployment (CD):** หาก Test ผ่าน 100% ระบบจะ Build Docker และทำการ Deploy อัปเดตไปยัง Hugging Face Spaces โดยอัตโนมัติ

## 🧪 การทดสอบระบบ (Testing)
โปรเจกต์นี้มี Artifacts สำหรับการทดสอบเตรียมไว้ในโฟลเดอร์ `api_testing/`:
* **Postman Collection:** `Dog-cat-api.postman_collection.json` สำหรับทดสอบฟังก์ชันการทำงาน
* **JMeter Test Plan:** ไฟล์ `.jmx` ทั้งสำหรับ Local และ Cloud สำหรับการทำ Load Testing ตามเอกสารประกอบการเรียน
