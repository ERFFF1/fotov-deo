#!/usr/bin/env python3
"""
Face/Video AI Studio - Başlangıç Scripti
Sistemi başlatmak için gerekli kontrolleri yapar
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_redis():
    """Redis'in çalışıp çalışmadığını kontrol et"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis bağlantısı başarılı")
        return True
    except Exception as e:
        print(f"❌ Redis bağlantısı başarısız: {e}")
        print("💡 Redis'i başlatmak için: redis-server")
        return False

def check_dependencies():
    """Gerekli bağımlılıkları kontrol et"""
    required_packages = [
        'fastapi', 'uvicorn', 'redis', 'rq', 'sqlalchemy', 
        'pydantic', 'pyyaml', 'opencv-python', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 Eksik paketleri kurmak için:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_directories():
    """Gerekli dizinleri oluştur"""
    directories = [
        "storage/uploads",
        "storage/outputs", 
        "storage/artifacts",
        "storage/cache",
        "db",
        "web/static/css",
        "web/static/js"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 {directory}")

def start_worker():
    """RQ worker'ı başlat"""
    print("\n🚀 RQ Worker başlatılıyor...")
    try:
        # Worker'ı arka planda başlat
        worker_process = subprocess.Popen([
            sys.executable, "-m", "workers.worker"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("✅ RQ Worker başlatıldı")
        return worker_process
    except Exception as e:
        print(f"❌ Worker başlatma hatası: {e}")
        return None

def start_web_server():
    """Web sunucusunu başlat"""
    print("\n🌐 Web sunucusu başlatılıyor...")
    print("🔗 WebSocket desteği aktif")
    print("📡 Gerçek zamanlı güncellemeler hazır")
    try:
        # FastAPI sunucusunu başlat
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload",
            "--ws-ping-interval", "30",
            "--ws-ping-timeout", "10"
        ])
    except KeyboardInterrupt:
        print("\n👋 Sunucu durduruldu")
    except Exception as e:
        print(f"❌ Sunucu başlatma hatası: {e}")

def main():
    """Ana başlangıç fonksiyonu"""
    print("🎯 Face/Video AI Studio Başlangıç Kontrolü")
    print("=" * 50)
    
    # 1. Bağımlılıkları kontrol et
    print("\n📦 Bağımlılık Kontrolü:")
    if not check_dependencies():
        print("\n❌ Gerekli bağımlılıklar eksik!")
        return False
    
    # 2. Redis'i kontrol et
    print("\n🔴 Redis Kontrolü:")
    if not check_redis():
        print("\n❌ Redis çalışmıyor!")
        print("💡 Redis'i başlatmak için ayrı bir terminalde: redis-server")
        return False
    
    # 3. Dizinleri oluştur
    print("\n📁 Dizin Yapısı:")
    setup_directories()
    
    # 4. Worker'ı başlat
    worker_process = start_worker()
    if not worker_process:
        return False
    
    # 5. Kısa bekleme
    print("\n⏳ Sistem hazırlanıyor...")
    time.sleep(2)
    
    # 6. Web sunucusunu başlat
    try:
        start_web_server()
    finally:
        # Worker'ı temizle
        if worker_process:
            worker_process.terminate()
            print("🧹 Worker temizlendi")

if __name__ == "__main__":
    main()
