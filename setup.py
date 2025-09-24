#!/usr/bin/env python3
"""
Fotoğraf ve Video İşleme Sistemi Kurulum Scripti
Bu script tüm gerekli kütüphaneleri kurar ve temel klasör yapısını oluşturur.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Komut çalıştır ve sonucu göster"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} tamamlandı!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} başarısız: {e}")
        print(f"Çıktı: {e.stdout}")
        print(f"Hata: {e.stderr}")
        return False

def create_directories():
    """Gerekli klasörleri oluştur"""
    directories = [
        "data/input",
        "data/output", 
        "data/models",
        "outputs/faces",
        "outputs/enhanced",
        "outputs/swap",
        "outputs/videos",
        "outputs/talking_heads",
        "work/frames",
        "work/temp"
    ]
    
    print("\n📁 Klasör yapısı oluşturuluyor...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")
    
    print("✅ Tüm klasörler oluşturuldu!")

def install_requirements():
    """Python kütüphanelerini kur"""
    print("\n📦 Python kütüphaneleri kuruluyor...")
    
    # pip'i güncelle
    run_command("python -m pip install --upgrade pip", "Pip güncelleniyor")
    
    # requirements.txt'den kur
    if Path("requirements.txt").exists():
        run_command("pip install -r requirements.txt", "Gerekli kütüphaneler kuruluyor")
    else:
        print("❌ requirements.txt bulunamadı!")
        return False
    
    return True

def download_models():
    """Gerekli AI modellerini indir"""
    print("\n🤖 AI modelleri indiriliyor...")
    
    # Haarcascade dosyalarını kontrol et
    import cv2
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if Path(cascade_path).exists():
        print("✅ Haarcascade yüz tespit modeli mevcut")
    else:
        print("❌ Haarcascade modeli bulunamadı!")
    
    print("ℹ️  Diğer modeller ilk kullanımda otomatik indirilecek")

def main():
    """Ana kurulum fonksiyonu"""
    print("🚀 Fotoğraf ve Video İşleme Sistemi Kurulumu Başlıyor...")
    print("=" * 60)
    
    # Python versiyonunu kontrol et
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ gerekli! Mevcut versiyon:", sys.version)
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor} tespit edildi")
    
    # Klasörleri oluştur
    create_directories()
    
    # Kütüphaneleri kur
    if not install_requirements():
        print("❌ Kurulum başarısız!")
        return False
    
    # Modelleri indir
    download_models()
    
    print("\n" + "=" * 60)
    print("🎉 Kurulum tamamlandı!")
    print("\n📋 Sonraki adımlar:")
    print("1. python face_detector.py [fotoğraf_yolu] - Yüz tespiti")
    print("2. python face_recognizer.py [foto1] [foto2] - Yüz tanıma")
    print("3. python face_swapper.py [kaynak] [hedef] - Yüz değiştirme")
    print("4. python video_processor.py [video] [yüz] - Video işleme")
    print("5. python talking_head.py [fotoğraf] [ses] - Konuşturma")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
