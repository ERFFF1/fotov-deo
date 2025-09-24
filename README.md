# 🎯 Face/Video AI Studio

**Yapay zeka destekli fotoğraf ve video işleme platformu**

## ✨ Özellikler

### 🎭 AI İşlemleri
- **Face Swap**: Yüz değiştirme (InsightFace)
- **Talking Head**: Konuşan kafa üretimi (Wav2Lip, SadTalker)
- **Face Enhancement**: Yüz iyileştirme (GFPGAN)
- **Color Grading**: Profesyonel renk düzeltme
- **Background Replacement**: AI destekli arka plan değiştirme
- **Video Post-Production**: Sinematik video işleme

### 🚀 Teknoloji Stack
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Modern HTML5 + CSS3 + JavaScript
- **Database**: SQLAlchemy + SQLite
- **Queue**: Redis + RQ
- **Real-time**: WebSocket
- **AI Models**: ONNX Runtime, PyTorch
- **Deployment**: Vercel Ready

### 🎨 Modern UI/UX
- **Smart Wizard**: Akıllı proje sihirbazı
- **Enhanced Gallery**: Gelişmiş galeri yönetimi
- **Comparison Mode**: Öncesi/sonrası karşılaştırma
- **Real-time Updates**: Canlı ilerleme takibi
- **Responsive Design**: Mobil uyumlu

### 🔒 Güvenlik & Etik
- **Consent Workflow**: 3 aşamalı rıza onayı
- **Automatic Watermarking**: Otomatik filigran
- **Audit Logging**: Şifreli audit logları
- **EXIF Metadata**: AI-Generated etiketleme

## 🛠️ Kurulum

### Yerel Geliştirme
```bash
# Repository'yi klonlayın
git clone https://github.com/yourusername/face-video-ai-studio.git
cd face-video-ai-studio

# Bağımlılıkları kurun
pip install -r requirements.txt

# Sistemi başlatın
python app.py
```

### Vercel Deployment
```bash
# Vercel CLI ile deploy
vercel --prod

# Veya GitHub'dan otomatik deploy
# Repository'yi Vercel'e bağlayın
```

## 📱 Kullanım

### Web Arayüzü
1. **Ana Sayfa**: http://localhost:8000
2. **Yeni Proje**: Akıllı sihirbaz ile proje oluşturun
3. **Galeri**: Sonuçları görüntüleyin ve yönetin
4. **API Docs**: http://localhost:8000/docs

### API Kullanımı
```python
import requests

# Yeni iş oluştur
response = requests.post('http://localhost:8000/api/jobs', json={
    "job_type": "face_swap",
    "inputs": {
        "source_image": "path/to/image.jpg",
        "target_video": "path/to/video.mp4"
    },
    "consent_tag": "demo"
})

job_id = response.json()['id']

# İş durumunu kontrol et
status = requests.get(f'http://localhost:8000/api/jobs/{job_id}')
```

## 🏗️ Mimari

### Core Modüller
- `core/config.py`: YAML konfigürasyon yönetimi
- `core/database.py`: SQLAlchemy veritabanı şeması
- `core/device_manager.py`: GPU/CPU kaynak yönetimi
- `core/ethics_security.py`: Etik ve güvenlik katmanı
- `core/websocket_manager.py`: Gerçek zamanlı iletişim
- `core/onnx_accelerator.py`: ONNX hızlandırma
- `core/metrics.py`: Nesnel kalite metrikleri

### Pipeline Modülleri
- `pipelines/face.py`: Yüz işleme pipeline'ı
- `pipelines/body.py`: Beden ve kıyafet işleme
- `pipelines/environment.py`: Arka plan ve çevre işleme
- `pipelines/color_audio.py`: Renk ve ses işleme
- `pipelines/talking.py`: Konuşan kafa üretimi
- `pipelines/advanced_ai.py`: Gelişmiş AI özellikleri
- `pipelines/temporal_consistency.py`: Zamansal tutarlılık
- `pipelines/post_process.py`: Post-prodüksiyon

## 📊 Performans

### Optimizasyonlar
- **ONNX Runtime**: CUDA/Metal/CPU desteği
- **Model Caching**: Otomatik model önbellekleme
- **GPU Management**: Merkezi GPU yönetimi
- **Parallel Processing**: Çoklu işlem desteği
- **Memory Optimization**: Bellek optimizasyonu

### Metrikler
- **Face Alignment Score**: Landmark reprojection error
- **Color ΔE**: CIELAB renk farkı
- **Temporal Jitter**: Optical flow standart sapması
- **Lip-sync LSE**: LSE-C/LSE-D metrikleri
- **PSNR/SSIM**: Görüntü kalite metrikleri

## 🔧 Konfigürasyon

### Preset'ler
- `face_swap_basic.yaml`: Temel yüz değiştirme
- `talking_wav2lip.yaml`: Wav2Lip konuşan kafa
- `color_grade_rec709.yaml`: Rec709 renk düzeltme

### Environment Variables
```bash
# Redis
REDIS_URL=redis://localhost:6379

# Database
DATABASE_URL=sqlite:///./db/app.db

# Storage
STORAGE_PATH=./storage

# Models
MODELS_PATH=./models
```

## 📈 Roadmap

### v1.1 (Gelecek)
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Advanced model management
- [ ] Batch processing API
- [ ] Webhook notifications

### v1.2 (Gelecek)
- [ ] Multi-tenant support
- [ ] Advanced analytics
- [ ] Plugin system
- [ ] Mobile app
- [ ] Enterprise features

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- [InsightFace](https://github.com/deepinsight/insightface) - Yüz tanıma
- [GFPGAN](https://github.com/TencentARC/GFPGAN) - Yüz iyileştirme
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) - Konuşan kafa
- [SadTalker](https://github.com/OpenTalker/SadTalker) - Konuşan kafa
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Vercel](https://vercel.com/) - Deployment platform

## 📞 İletişim

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Email**: your.email@example.com
- **Website**: https://yourwebsite.com

---

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**