# 🚀 Face/Video AI Studio - Deployment Guide

## 📋 **Vercel + Worker Mimarisi**

### **🏗️ Mimari:**
- **Vercel**: Hafif web arayüzü ve API
- **Worker**: Ağır ML işlemleri (Railway/Fly.io/EC2)
- **Redis**: Upstash (serverless Redis)

---

## ⚙️ **1. Vercel Deployment**

### **Konfigürasyon:**
- **Framework**: Other
- **Root Directory**: `./`
- **Build Command**: `pip install -r requirements-vercel.txt`
- **Output Directory**: `./`
- **Install Command**: `pip install -r requirements-vercel.txt`

### **Environment Variables:**
```bash
REDIS_URL=redis://your-upstash-url
```

### **Dosyalar:**
- ✅ `requirements-vercel.txt` - Hafif bağımlılıklar
- ✅ `vercel.json` - Functions konfigürasyonu
- ✅ `api/index.py` - Giriş noktası
- ✅ `.vercelignore` - Büyük dosyaları dışla

---

## 🔧 **2. Worker Deployment**

### **Railway/Fly.io/EC2:**
```bash
# Worker bağımlılıkları
pip install -r workers/requirements-worker.txt

# Worker başlatma
python workers/worker.py
```

### **Environment Variables:**
```bash
REDIS_URL=redis://your-upstash-url
```

---

## 📊 **3. Redis Setup (Upstash)**

### **Upstash Redis:**
1. https://upstash.com adresine gidin
2. Yeni Redis database oluşturun
3. Connection string'i alın
4. Vercel ve Worker'da `REDIS_URL` olarak ayarlayın

---

## 🎯 **4. Test Deployment**

### **Vercel Test:**
```bash
curl https://your-app.vercel.app/api/health
```

### **Worker Test:**
```bash
# Redis'e test job ekle
rq enqueue workers.tasks.process_job_task {
  "job_type": "face_swap",
  "inputs": {"source_image": "test.jpg"},
  "params": {"quality": "high"}
}
```

---

## 📈 **5. Monitoring**

### **Vercel:**
- Dashboard'da deployment durumu
- Function logs
- Performance metrics

### **Worker:**
- RQ dashboard
- Redis monitoring
- Error logs

---

## 🔄 **6. Workflow**

1. **Kullanıcı** → Vercel web arayüzü
2. **Vercel** → Redis'e job ekler
3. **Worker** → Redis'ten job alır
4. **Worker** → ML işlemi yapar
5. **Worker** → Sonucu Redis'e yazar
6. **Vercel** → Kullanıcıya sonucu gösterir

---

## ⚠️ **Önemli Notlar**

### **Vercel Limitleri:**
- **Function Timeout**: 10 saniye
- **Memory**: 1024 MB
- **Build Timeout**: 5 dakika

### **Worker Gereksinimleri:**
- **GPU**: CUDA desteği (opsiyonel)
- **Memory**: 8GB+ önerilen
- **Storage**: Model dosyaları için

### **Redis:**
- **Upstash**: Serverless, Vercel uyumlu
- **Connection Limit**: 1000 concurrent
- **Memory**: 1GB+ önerilen

---

## 🎉 **Sonuç**

**Bu mimari ile:**
- ✅ **Vercel**: Hızlı web arayüzü
- ✅ **Worker**: Güçlü ML işlemleri
- ✅ **Redis**: Güvenilir kuyruk sistemi
- ✅ **Scalable**: Otomatik ölçeklendirme

**Artık AI Studio'nuz production-ready!** 🚀
