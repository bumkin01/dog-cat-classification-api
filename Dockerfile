FROM python:3.9-slim

WORKDIR /app

# ติดตั้ง System dependencies ที่จำเป็นสำหรับประมวลผลรูป 
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev zlib1g-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*
    
# ติดตั้ง Python Libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# นำโค้ดทั้งหมดเข้าโฟลเดอร์ทำงาน
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]