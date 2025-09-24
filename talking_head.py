#!/usr/bin/env python3
"""
Gelişmiş Konuşan Kafa (Talking Head) Sistemi
Wav2Lip ve SadTalker entegrasyonu ile fotoğrafları konuşturur
"""

import cv2
import numpy as np
import sys
import os
import subprocess
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import tempfile
import shutil
import requests
from tqdm import tqdm

class TalkingHeadGenerator:
    def __init__(self, method='wav2lip'):
        """
        Konuşan kafa üretici başlatıcı
        
        Args:
            method (str): 'wav2lip', 'sadtalker' seçenekleri
        """
        self.method = method
        self.setup_generator()
        
    def setup_generator(self):
        """Seçilen yönteme göre üretici sistemini kur"""
        if self.method == 'wav2lip':
            self.setup_wav2lip()
        elif self.method == 'sadtalker':
            self.setup_sadtalker()
    
    def setup_wav2lip(self):
        """Wav2Lip sistemini kur"""
        try:
            # Wav2Lip klasörünü kontrol et
            wav2lip_dir = Path("models/wav2lip")
            if not wav2lip_dir.exists():
                print("📥 Wav2Lip indiriliyor...")
                self.download_wav2lip()
            
            # Model dosyasını kontrol et
            model_path = wav2lip_dir / "Wav2Lip.pth"
            if not model_path.exists():
                print("📥 Wav2Lip modeli indiriliyor...")
                self.download_wav2lip_model()
            
            print("✅ Wav2Lip sistemi hazır")
            
        except Exception as e:
            print(f"❌ Wav2Lip kurulum hatası: {e}")
            print("⚠️  SadTalker'a geçiliyor...")
            self.method = 'sadtalker'
            self.setup_sadtalker()
    
    def setup_sadtalker(self):
        """SadTalker sistemini kur"""
        try:
            # SadTalker klasörünü kontrol et
            sadtalker_dir = Path("models/sadtalker")
            if not sadtalker_dir.exists():
                print("📥 SadTalker indiriliyor...")
                self.download_sadtalker()
            
            print("✅ SadTalker sistemi hazır")
            
        except Exception as e:
            print(f"❌ SadTalker kurulum hatası: {e}")
            print("⚠️  Basit OpenCV yöntemine geçiliyor...")
            self.method = 'opencv'
    
    def download_wav2lip(self):
        """Wav2Lip repository'sini indir"""
        try:
            wav2lip_dir = Path("models/wav2lip")
            wav2lip_dir.mkdir(parents=True, exist_ok=True)
            
            # Git clone komutu
            cmd = [
                'git', 'clone', 
                'https://github.com/Rudrabha/Wav2Lip.git',
                str(wav2lip_dir)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            print("✅ Wav2Lip indirildi")
            
        except Exception as e:
            print(f"❌ Wav2Lip indirme hatası: {e}")
            raise
    
    def download_wav2lip_model(self):
        """Wav2Lip modelini indir"""
        try:
            model_url = "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIYOXefzlSSW0WFlFR_6UDg?e=n9ljGW&download=1"
            model_path = Path("models/wav2lip/Wav2Lip.pth")
            
            print("📥 Model indiriliyor (bu işlem biraz zaman alabilir)...")
            
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Model indiriliyor") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print("✅ Wav2Lip modeli indirildi")
            
        except Exception as e:
            print(f"❌ Model indirme hatası: {e}")
            raise
    
    def download_sadtalker(self):
        """SadTalker repository'sini indir"""
        try:
            sadtalker_dir = Path("models/sadtalker")
            sadtalker_dir.mkdir(parents=True, exist_ok=True)
            
            # Git clone komutu
            cmd = [
                'git', 'clone', 
                'https://github.com/Winfredy/SadTalker.git',
                str(sadtalker_dir)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            print("✅ SadTalker indirildi")
            
        except Exception as e:
            print(f"❌ SadTalker indirme hatası: {e}")
            raise
    
    def preprocess_image_for_wav2lip(self, image_path: str, output_path: str) -> bool:
        """Wav2Lip için görüntüyü ön işle"""
        try:
            # Görüntüyü yükle
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Yüz tespiti
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                print("⚠️  Görüntüde yüz bulunamadı")
                return False
            
            # En büyük yüzü al
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Yüzü kırp ve 512x512'e yeniden boyutlandır
            face_crop = image[y:y+h, x:x+w]
            resized_face = cv2.resize(face_crop, (512, 512))
            
            # Kaydet
            cv2.imwrite(output_path, resized_face)
            return True
            
        except Exception as e:
            print(f"❌ Görüntü ön işleme hatası: {e}")
            return False
    
    def generate_talking_head_wav2lip(self, image_path: str, audio_path: str, output_path: str) -> Dict:
        """Wav2Lip ile konuşan kafa üret"""
        try:
            wav2lip_dir = Path("models/wav2lip")
            model_path = wav2lip_dir / "Wav2Lip.pth"
            
            if not model_path.exists():
                return {
                    'success': False,
                    'error': 'Wav2Lip modeli bulunamadı',
                    'output_path': None
                }
            
            # Geçici dosyalar
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_image = Path(temp_dir) / "processed_face.jpg"
                
                # Görüntüyü ön işle
                if not self.preprocess_image_for_wav2lip(image_path, str(temp_image)):
                    return {
                        'success': False,
                        'error': 'Görüntü ön işleme başarısız',
                        'output_path': None
                    }
                
                # Wav2Lip inference komutu
                cmd = [
                    'python', str(wav2lip_dir / "inference.py"),
                    '--checkpoint_path', str(model_path),
                    '--face', str(temp_image),
                    '--audio', audio_path,
                    '--outfile', output_path
                ]
                
                print(f"🎬 Wav2Lip ile konuşan kafa üretiliyor...")
                print(f"   Görüntü: {image_path}")
                print(f"   Ses: {audio_path}")
                print(f"   Çıktı: {output_path}")
                
                # Komutu çalıştır
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=wav2lip_dir)
                
                if result.returncode == 0:
                    print("✅ Wav2Lip konuşan kafa üretimi tamamlandı")
                    
                    return {
                        'success': True,
                        'method': 'wav2lip',
                        'input_image': image_path,
                        'input_audio': audio_path,
                        'output_path': output_path,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Wav2Lip hatası: {result.stderr}',
                        'output_path': None
                    }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'Wav2Lip üretim hatası: {e}',
                'output_path': None
            }
    
    def generate_talking_head_sadtalker(self, image_path: str, audio_path: str, output_path: str) -> Dict:
        """SadTalker ile konuşan kafa üret"""
        try:
            sadtalker_dir = Path("models/sadtalker")
            
            if not sadtalker_dir.exists():
                return {
                    'success': False,
                    'error': 'SadTalker bulunamadı',
                    'output_path': None
                }
            
            # SadTalker inference komutu
            cmd = [
                'python', str(sadtalker_dir / "inference.py"),
                '--driven_audio', audio_path,
                '--source_image', image_path,
                '--enhancer', 'gfpgan',
                '--result_dir', str(Path(output_path).parent),
                '--still',  # Sadece kafa hareketi
                '--preprocess', 'full'  # Tam ön işleme
            ]
            
            print(f"🎬 SadTalker ile konuşan kafa üretiliyor...")
            print(f"   Görüntü: {image_path}")
            print(f"   Ses: {audio_path}")
            print(f"   Çıktı: {output_path}")
            
            # Komutu çalıştır
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=sadtalker_dir)
            
            if result.returncode == 0:
                # SadTalker çıktı dosyasını bul
                result_dir = Path(output_path).parent
                sadtalker_output = None
                
                for file in result_dir.glob("*.mp4"):
                    if "sadtalker" in file.name.lower():
                        sadtalker_output = file
                        break
                
                if sadtalker_output and sadtalker_output.exists():
                    # Çıktı dosyasını istenen konuma taşı
                    shutil.move(str(sadtalker_output), output_path)
                    
                    print("✅ SadTalker konuşan kafa üretimi tamamlandı")
                    
                    return {
                        'success': True,
                        'method': 'sadtalker',
                        'input_image': image_path,
                        'input_audio': audio_path,
                        'output_path': output_path,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'success': False,
                        'error': 'SadTalker çıktı dosyası bulunamadı',
                        'output_path': None
                    }
            else:
                return {
                    'success': False,
                    'error': f'SadTalker hatası: {result.stderr}',
                    'output_path': None
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'SadTalker üretim hatası: {e}',
                'output_path': None
            }
    
    def generate_talking_head_opencv(self, image_path: str, audio_path: str, output_path: str) -> Dict:
        """OpenCV ile basit konuşan kafa üret (demo)"""
        try:
            # Görüntüyü yükle
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': f'Görüntü yüklenemedi: {image_path}',
                    'output_path': None
                }
            
            # Yüz tespiti
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {
                    'success': False,
                    'error': 'Görüntüde yüz bulunamadı',
                    'output_path': None
                }
            
            # En büyük yüzü al
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Basit animasyon (yüzü hafifçe hareket ettir)
            frames = []
            fps = 25
            duration = 3  # 3 saniye
            
            for i in range(fps * duration):
                # Yüzü hafifçe hareket ettir
                offset_x = int(5 * np.sin(2 * np.pi * i / fps))
                offset_y = int(3 * np.cos(2 * np.pi * i / fps))
                
                # Yeni pozisyon
                new_x = max(0, x + offset_x)
                new_y = max(0, y + offset_y)
                new_w = min(image.shape[1] - new_x, w)
                new_h = min(image.shape[0] - new_y, h)
                
                # Frame oluştur
                frame = image.copy()
                
                # Yüz bölgesini vurgula
                cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
                
                # "Konuşuyor" efekti (basit)
                if i % 10 < 5:  # Yanıp sönen efekt
                    cv2.circle(frame, (new_x + new_w//2, new_y + new_h//2), 5, (0, 0, 255), -1)
                
                frames.append(frame)
            
            # Video oluştur
            height, width, _ = image.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            
            print("✅ OpenCV konuşan kafa üretimi tamamlandı (demo)")
            
            return {
                'success': True,
                'method': 'opencv_demo',
                'input_image': image_path,
                'input_audio': audio_path,
                'output_path': output_path,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'OpenCV üretim hatası: {e}',
                'output_path': None
            }
    
    def generate_talking_head(self, image_path: str, audio_path: str, output_path: Optional[str] = None) -> Dict:
        """Ana konuşan kafa üretim fonksiyonu"""
        # Dosya kontrolü
        if not os.path.exists(image_path):
            return {
                'success': False,
                'error': f'Görüntü dosyası bulunamadı: {image_path}',
                'output_path': None
            }
        
        if not os.path.exists(audio_path):
            return {
                'success': False,
                'error': f'Ses dosyası bulunamadı: {audio_path}',
                'output_path': None
            }
        
        # Çıktı dosya yolu belirle
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"talking_head_{self.method}_{timestamp}.mp4"
            output_path = f"outputs/talking_heads/{filename}"
        
        # Çıktı klasörünü oluştur
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"🎬 Konuşan kafa üretimi başlıyor...")
        print(f"   Yöntem: {self.method}")
        print(f"   Görüntü: {image_path}")
        print(f"   Ses: {audio_path}")
        print(f"   Çıktı: {output_path}")
        
        # Yönteme göre üretim yap
        if self.method == 'wav2lip':
            return self.generate_talking_head_wav2lip(image_path, audio_path, output_path)
        elif self.method == 'sadtalker':
            return self.generate_talking_head_sadtalker(image_path, audio_path, output_path)
        elif self.method == 'opencv':
            return self.generate_talking_head_opencv(image_path, audio_path, output_path)
        else:
            return {
                'success': False,
                'error': f'Bilinmeyen yöntem: {self.method}',
                'output_path': None
            }
    
    def batch_generate_talking_heads(self, images_dir: str, audio_path: str, output_dir: str = "outputs/talking_heads_batch") -> Dict:
        """Birden fazla görüntü için toplu konuşan kafa üretimi"""
        if not os.path.exists(images_dir):
            return {
                'success': False,
                'error': f'Görüntü klasörü bulunamadı: {images_dir}',
                'processed_count': 0
            }
        
        if not os.path.exists(audio_path):
            return {
                'success': False,
                'error': f'Ses dosyası bulunamadı: {audio_path}',
                'processed_count': 0
            }
        
        # Desteklenen dosya formatları
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Görüntü dosyalarını bul
        image_files = [f for f in Path(images_dir).iterdir() 
                      if f.suffix.lower() in supported_formats]
        
        if not image_files:
            return {
                'success': False,
                'error': f'Görüntü dosyası bulunamadı: {images_dir}',
                'processed_count': 0
            }
        
        print(f"🔄 {len(image_files)} görüntü için toplu konuşan kafa üretimi...")
        
        # Çıktı klasörünü oluştur
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        success_count = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"   [{i}/{len(image_files)}] {image_file.name}")
            
            # Çıktı dosya yolu
            output_file = output_path / f"talking_{image_file.stem}.mp4"
            
            # Konuşan kafa üret
            result = self.generate_talking_head(str(image_file), audio_path, str(output_file))
            results.append(result)
            
            if result['success']:
                success_count += 1
        
        print(f"✅ Toplu konuşan kafa üretimi tamamlandı: {success_count}/{len(image_files)} başarılı")
        
        return {
            'success': True,
            'images_dir': images_dir,
            'audio_path': audio_path,
            'output_dir': output_dir,
            'total_files': len(image_files),
            'processed_count': success_count,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

def main():
    parser = argparse.ArgumentParser(description='Gelişmiş Konuşan Kafa Üretim Sistemi')
    parser.add_argument('--image', help='Kaynak görüntü dosyası')
    parser.add_argument('--audio', help='Ses dosyası (wav/mp3)')
    parser.add_argument('--method', choices=['wav2lip', 'sadtalker', 'opencv'], 
                       default='wav2lip', help='Üretim yöntemi')
    parser.add_argument('--output', help='Çıktı video yolu')
    parser.add_argument('--batch', action='store_true', 
                       help='Toplu üretim (image klasör olmalı)')
    parser.add_argument('--images-dir', help='Görüntü klasörü (batch modu için)')
    
    args = parser.parse_args()
    
    # Konuşan kafa üretici oluştur
    generator = TalkingHeadGenerator(method=args.method)
    
    if args.batch and args.images_dir and args.audio:
        # Toplu üretim
        result = generator.batch_generate_talking_heads(args.images_dir, args.audio, args.output)
        
        if result['success']:
            print(f"\n📊 Toplu Üretim Sonuçları:")
            print(f"   Görüntü klasörü: {result['images_dir']}")
            print(f"   Ses dosyası: {result['audio_path']}")
            print(f"   Toplam dosya: {result['total_files']}")
            print(f"   İşlenen: {result['processed_count']}")
            print(f"   Çıktı klasörü: {result['output_dir']}")
        else:
            print(f"❌ Hata: {result['error']}")
    
    elif args.image and args.audio:
        # Tek dosya üretim
        result = generator.generate_talking_head(args.image, args.audio, args.output)
        
        if result['success']:
            print(f"\n📊 Konuşan Kafa Üretim Sonuçları:")
            print(f"   Yöntem: {result['method']}")
            print(f"   Görüntü: {result['input_image']}")
            print(f"   Ses: {result['input_audio']}")
            print(f"   Çıktı: {result['output_path']}")
        else:
            print(f"❌ Hata: {result['error']}")
    
    else:
        print("❌ Lütfen gerekli parametreleri belirtin. --help ile yardım alın.")
        print("\nÖrnek kullanımlar:")
        print("  python talking_head.py --image face.jpg --audio voice.wav")
        print("  python talking_head.py --images-dir photos/ --audio voice.wav --batch")
        print("  python talking_head.py --image face.jpg --audio voice.wav --method sadtalker")

if __name__ == "__main__":
    main()
