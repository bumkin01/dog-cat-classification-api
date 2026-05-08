from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import os
import io
import sys
from PIL import Image, UnidentifiedImageError
import asyncio
from concurrent.futures import ProcessPoolExecutor

app = FastAPI(title="ML Model API - Production Ready")

# กำหนดขนาดไฟล์สูงสุด (เช่น 5MB)
MAX_FILE_SIZE = 5 * 1024 * 1024 

# ตัวแปรสำหรับเก็บ Model Session ภายในแต่ละ Worker Process
worker_session = None

def init_worker():
    """ฟังก์ชันนี้จะถูกเรียกตอนสร้าง Worker Process ใหม่ เพื่อโหลดโมเดลเตรียมไว้ในหน่วยความจำของ Process นั้น"""
    global worker_session

    # 1. ใช้ abspath ป้องกันปัญหา Relative Path เพี้ยนใน Docker
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "model", "hf_resnet50_prep.onnx")
    try:
        sess_options = ort.SessionOptions()
        # จำกัดให้ 1 Worker ใช้ CPU แค่ 2 Thread
        sess_options.intra_op_num_threads = 2 
        sess_options.inter_op_num_threads = 2

        # 2. ดักจับและเช็คให้ชัดเจนว่าไฟล์มีอยู่จริงใน Docker หรือไม่
        if not os.path.exists(MODEL_PATH):
            print(f"[ERROR] ❌ ไม่พบไฟล์โมเดลที่ Path: {MODEL_PATH}", flush=True)
            return # หยุดการทำงาน
            
        worker_session = ort.InferenceSession(MODEL_PATH, sess_options=sess_options,providers=['CPUExecutionProvider'])
        print(f"[SUCCESS] ✅ โหลดโมเดลใน Worker สำเร็จจาก: {MODEL_PATH}", flush=True)
        
    except Exception as e:
        print(f"[ERROR] ❌ ONNX โหลดโมเดลไม่สำเร็จ สาเหตุ: {e}", flush=True)

# สร้าง Process Pool 
executor = ProcessPoolExecutor(max_workers=2, initializer=init_worker)

class PredictionResponse(BaseModel):
    prediction_class: str
    confidence: float

def map_imagenet_to_dogcat(class_id: int) -> str:
    """แปลงรหัส ImageNet 1000 คลาส ให้เหลือแค่ Dog, Cat, Other"""
    # ใน ImageNet หมาจะอยู่รหัส 151 ถึง 268
    if 151 <= class_id <= 268:
        return "Dog"
    # แมวจะอยู่รหัส 281 ถึง 285
    elif 281 <= class_id <= 285:
        return "Cat"
    else:
        return "Other (Unknown)"

def run_inference(image_bytes: bytes) -> tuple:
    """ฟังก์ชันรันโมเดลที่จะถูกส่งไปทำใน Worker Process"""
    global worker_session
    if worker_session is None:
        raise RuntimeError("Model not loaded in worker process")
    
    # Preprocessing
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img_data = np.array(img).astype('float32')
    
    mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    std = np.array([0.229, 0.224, 0.225]).astype('float32')
    img_data = (img_data / 255.0 - mean) / std
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    
    # Inference
    input_name = worker_session.get_inputs()[0].name
    result = worker_session.run(None, {input_name: img_data})
    output = result[0][0] 
    
    # Softmax
    exp_preds = np.exp(output - np.max(output))
    softmax_preds = exp_preds / np.sum(exp_preds)
    pred_class = int(np.argmax(softmax_preds))
    confidence = float(softmax_preds[pred_class])
    
    # แปลงตัวเลขเป็นตัวหนังสือ
    final_label = map_imagenet_to_dogcat(pred_class)
    
    return final_label, confidence

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # 1. เช็คประเภทไฟล์ (MIME Type) - ป้องกันอัปโหลด PDF, TXT
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Please upload an image.")
    
    # 2. เช็คขนาดไฟล์ - ป้องกัน Out of Memory (OOM)
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail="File too large. Maximum size is 5MB.")
    
    # 3. เช็คไฟล์รูปเสีย (Corrupted Image)
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify() # เช็คความสมบูรณ์โดยไม่ต้องถอดรหัสภาพทั้งรูป
    except UnidentifiedImageError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Corrupted or invalid image file.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Image processing error: {str(e)}")

    # 4. โยนงานไปให้ ProcessPool ทำงานเพื่อไม่ให้ API ค้าง (Non-blocking)
    loop = asyncio.get_running_loop()
    try:
        pred_class, confidence = await loop.run_in_executor(executor, run_inference, image_bytes)
        return {"prediction_class": pred_class, "confidence": confidence}
    except RuntimeError as e:
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {str(e)}")