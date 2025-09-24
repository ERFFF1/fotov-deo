#!/usr/bin/env python3
"""
Gelişmiş Yüz İyileştirme ve Düzenleme Sistemi
GFPGAN, Real-ESRGAN ve OpenCV kullanarak yüz kalitesini artırır
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Optional, Tuple, List, Dict

class FaceEnhancer:
    def __init__(self, method='gfpgan'):
        """
        Yüz iyileştirme sistemi başlatıcı
        
        Args:
            method (str): 'gfpgan', 'real_esrgan', 'opencv' seçenekleri
        """
        self.method = method
        self.setup_enhancer()
        
    def setup_enhancer(self):
        """Seçilen yönteme göre iyileştirme sistemini kur"""
        if self.method == 'gfpgan':
            try:
                from gfpgan import GFPGANer
                self.enhancer = GFPGANer(
                    model_path=None,  # Otomatik indir
                    upscale=2,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None
                )
                print("✅ GFPGAN iyileştirme sistemi hazır")
            except ImportError:
                print("❌ GFPGAN kurulu değil, OpenCV'ye geçiliyor")
                self.method = 'opencv'
                self.setup_enhancer()
                
        elif self.method == 'real_esrgan':
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                from basicsr.archs.srvgg_arch import SRVGGNetCompact
                
                # Real-ESRGAN modeli
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                      num_conv=32, upscale=4, act_type='prelu')
                self.enhancer = RealESRGANer(
                    scale=4,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=True
                )
                print("✅ Real-ESRGAN iyileştirme sistemi hazır")
            except ImportError:
                print("❌ Real-ESRGAN kurulu değil, OpenCV'ye geçiliyor")
                self.method = 'opencv'
                self.setup_enhancer()
                
        elif self.method == 'opencv':
            print("✅ OpenCV iyileştirme sistemi hazır")
    
    def enhance_face_gfpgan(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """GFPGAN ile yüz iyileştirme"""
        try:
            # GFPGAN iyileştirme
            _, _, enhanced_img = self.enhancer.enhance(
                image, 
                has_aligned=False, 
                only_center_face=True, 
                paste_back=True
            )
            
            # Kalite skoru hesapla (basit bir metrik)
            quality_score = self.calculate_quality_score(enhanced_img)
            
            return enhanced_img, quality_score
            
        except Exception as e:
            print(f"❌ GFPGAN hatası: {e}")
            return image, 0.0
    
    def enhance_face_real_esrgan(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Real-ESRGAN ile yüz iyileştirme"""
        try:
            # Real-ESRGAN iyileştirme
            enhanced_img, _ = self.enhancer.enhance(image, outscale=4)
            
            # Orijinal boyuta geri döndür
            h, w = image.shape[:2]
            enhanced_img = cv2.resize(enhanced_img, (w, h))
            
            # Kalite skoru hesapla
            quality_score = self.calculate_quality_score(enhanced_img)
            
            return enhanced_img, quality_score
            
        except Exception as e:
            print(f"❌ Real-ESRGAN hatası: {e}")
            return image, 0.0
    
    def enhance_face_opencv(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """OpenCV ile temel yüz iyileştirme"""
        try:
            # Gürültü azaltma
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            
            # Keskinlik artırma
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # Kontrast ve parlaklık ayarı
            enhanced = cv2.convertScaleAbs(sharpened, alpha=1.2, beta=10)
            
            # Kalite skoru hesapla
            quality_score = self.calculate_quality_score(enhanced)
            
            return enhanced, quality_score
            
        except Exception as e:
            print(f"❌ OpenCV hatası: {e}")
            return image, 0.0
    
    def calculate_quality_score(self, image: np.ndarray) -> float:
        """Görüntü kalite skoru hesapla (0-1 arası)"""
        try:
            # Laplacian varyansı ile keskinlik ölçümü
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize et (0-1 arası)
            quality_score = min(laplacian_var / 1000.0, 1.0)
            
            return quality_score
            
        except Exception as e:
            print(f"⚠️  Kalite skoru hesaplanamadı: {e}")
            return 0.5
    
    def enhance_face(self, image_path: str, output_path: Optional[str] = None) -> Dict:
        """Ana yüz iyileştirme fonksiyonu"""
        if not os.path.exists(image_path):
            return {
                'success': False,
                'error': f'Dosya bulunamadı: {image_path}',
                'output_path': None,
                'quality_score': 0.0
            }
        
        # Görüntüyü yükle
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'error': f'Görüntü yüklenemedi: {image_path}',
                'output_path': None,
                'quality_score': 0.0
            }
        
        print(f"🔧 {self.method.upper()} ile yüz iyileştirme yapılıyor...")
        
        # Yönteme göre iyileştirme yap
        if self.method == 'gfpgan':
            enhanced_image, quality_score = self.enhance_face_gfpgan(image)
        elif self.method == 'real_esrgan':
            enhanced_image, quality_score = self.enhance_face_real_esrgan(image)
        elif self.method == 'opencv':
            enhanced_image, quality_score = self.enhance_face_opencv(image)
        
        # Çıktı dosya yolu belirle
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_{self.method}_{timestamp}.jpg"
            output_path = f"outputs/enhanced/{filename}"
        
        # Çıktı klasörünü oluştur
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # İyileştirilmiş görüntüyü kaydet
        success = cv2.imwrite(output_path, enhanced_image)
        
        if success:
            print(f"✅ İyileştirme tamamlandı!")
            print(f"   Kalite skoru: {quality_score:.3f}")
            print(f"   Çıktı: {output_path}")
            
            return {
                'success': True,
                'input_path': image_path,
                'output_path': output_path,
                'method': self.method,
                'quality_score': float(quality_score),
                'original_size': image.shape[:2],
                'enhanced_size': enhanced_image.shape[:2],
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'success': False,
                'error': 'İyileştirilmiş görüntü kaydedilemedi',
                'output_path': None,
                'quality_score': float(quality_score)
            }
    
    def batch_enhance(self, input_dir: str, output_dir: str = "outputs/enhanced_batch") -> Dict:
        """Klasördeki tüm görüntüleri toplu iyileştir"""
        input_path = Path(input_dir)
        if not input_path.exists():
            return {
                'success': False,
                'error': f'Giriş klasörü bulunamadı: {input_dir}',
                'processed_count': 0
            }
        
        # Desteklenen dosya formatları
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Görüntü dosyalarını bul
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in supported_formats]
        
        if not image_files:
            return {
                'success': False,
                'error': f'Görüntü dosyası bulunamadı: {input_dir}',
                'processed_count': 0
            }
        
        print(f"🔄 {len(image_files)} dosya toplu iyileştiriliyor...")
        
        # Çıktı klasörünü oluştur
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        success_count = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"   [{i}/{len(image_files)}] {image_file.name}")
            
            # Çıktı dosya yolu
            output_file = output_path / f"enhanced_{image_file.name}"
            
            # İyileştirme yap
            result = self.enhance_face(str(image_file), str(output_file))
            results.append(result)
            
            if result['success']:
                success_count += 1
        
        print(f"✅ Toplu iyileştirme tamamlandı: {success_count}/{len(image_files)} başarılı")
        
        return {
            'success': True,
            'input_dir': input_dir,
            'output_dir': output_dir,
            'total_files': len(image_files),
            'processed_count': success_count,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def compare_enhancement_methods(self, image_path: str) -> Dict:
        """Farklı iyileştirme yöntemlerini karşılaştır"""
        if not os.path.exists(image_path):
            return {
                'success': False,
                'error': f'Dosya bulunamadı: {image_path}',
                'comparison': {}
            }
        
        methods = ['opencv', 'gfpgan', 'real_esrgan']
        comparison_results = {}
        
        print(f"🔍 Farklı yöntemlerle iyileştirme karşılaştırması...")
        
        for method in methods:
            print(f"   {method.upper()} yöntemi test ediliyor...")
            
            # Geçici olarak yöntemi değiştir
            original_method = self.method
            self.method = method
            self.setup_enhancer()
            
            # İyileştirme yap
            result = self.enhance_face(image_path)
            
            if result['success']:
                comparison_results[method] = {
                    'quality_score': result['quality_score'],
                    'output_path': result['output_path'],
                    'success': True
                }
            else:
                comparison_results[method] = {
                    'error': result['error'],
                    'success': False
                }
            
            # Orijinal yöntemi geri yükle
            self.method = original_method
            self.setup_enhancer()
        
        # En iyi yöntemi belirle
        best_method = None
        best_score = 0.0
        
        for method, result in comparison_results.items():
            if result['success'] and result['quality_score'] > best_score:
                best_score = result['quality_score']
                best_method = method
        
        return {
            'success': True,
            'input_path': image_path,
            'comparison': comparison_results,
            'best_method': best_method,
            'best_score': best_score,
            'timestamp': datetime.now().isoformat()
        }

def main():
    parser = argparse.ArgumentParser(description='Gelişmiş Yüz İyileştirme Sistemi')
    parser.add_argument('input_path', help='İyileştirilecek görüntü yolu veya klasör')
    parser.add_argument('--method', choices=['gfpgan', 'real_esrgan', 'opencv'], 
                       default='gfpgan', help='İyileştirme yöntemi')
    parser.add_argument('--output', help='Çıktı dosya/klasör yolu')
    parser.add_argument('--batch', action='store_true', 
                       help='Klasördeki tüm görüntüleri toplu iyileştir')
    parser.add_argument('--compare', action='store_true',
                       help='Farklı yöntemleri karşılaştır')
    
    args = parser.parse_args()
    
    # İyileştirme sistemi oluştur
    enhancer = FaceEnhancer(method=args.method)
    
    if args.compare:
        # Yöntem karşılaştırması
        result = enhancer.compare_enhancement_methods(args.input_path)
        
        if result['success']:
            print(f"\n📊 Yöntem Karşılaştırması:")
            print(f"   En iyi yöntem: {result['best_method']} (Skor: {result['best_score']:.3f})")
            
            print(f"\n📋 Tüm sonuçlar:")
            for method, data in result['comparison'].items():
                if data['success']:
                    print(f"   {method}: {data['quality_score']:.3f}")
                else:
                    print(f"   {method}: Hata - {data['error']}")
            
            # JSON raporu kaydet
            json_path = f"outputs/enhancement_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"📄 Detaylı rapor: {json_path}")
        else:
            print(f"❌ Hata: {result['error']}")
    
    elif args.batch:
        # Toplu iyileştirme
        result = enhancer.batch_enhance(args.input_path, args.output)
        
        if result['success']:
            print(f"\n📊 Toplu İyileştirme Sonuçları:")
            print(f"   Toplam dosya: {result['total_files']}")
            print(f"   İşlenen: {result['processed_count']}")
            print(f"   Çıktı klasörü: {result['output_dir']}")
        else:
            print(f"❌ Hata: {result['error']}")
    
    else:
        # Tek dosya iyileştirme
        result = enhancer.enhance_face(args.input_path, args.output)
        
        if result['success']:
            print(f"\n📊 İyileştirme Sonuçları:")
            print(f"   Yöntem: {result['method']}")
            print(f"   Kalite skoru: {result['quality_score']:.3f}")
            print(f"   Orijinal boyut: {result['original_size']}")
            print(f"   İyileştirilmiş boyut: {result['enhanced_size']}")
            print(f"   Çıktı: {result['output_path']}")
        else:
            print(f"❌ Hata: {result['error']}")

if __name__ == "__main__":
    main()
