#!/usr/bin/env python3
"""
Gelişmiş Yüz Değiştirme (Face Swap) Sistemi
InsightFace ve OpenCV kullanarak gerçekçi yüz değiştirme yapar
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
import warnings
warnings.filterwarnings('ignore')

class FaceSwapper:
    def __init__(self, method='insightface'):
        """
        Yüz değiştirme sistemi başlatıcı
        
        Args:
            method (str): 'insightface', 'opencv' seçenekleri
        """
        self.method = method
        self.swapper_model = None
        self.setup_swapper()
        
    def setup_swapper(self):
        """Seçilen yönteme göre değiştirme sistemini kur"""
        if self.method == 'insightface':
            try:
                import insightface
                from insightface.app import FaceAnalysis
                
                # Yüz analiz modeli
                self.face_analyzer = FaceAnalysis(name='buffalo_l')
                self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
                
                # Yüz değiştirme modeli
                self.swapper_model = insightface.model_zoo.get_model(
                    'inswapper_128.onnx', 
                    download=True, 
                    download_zip=True
                )
                
                print("✅ InsightFace yüz değiştirme sistemi hazır")
                
            except ImportError:
                print("❌ InsightFace kurulu değil, OpenCV'ye geçiliyor")
                self.method = 'opencv'
                self.setup_swapper()
                
        elif self.method == 'opencv':
            # OpenCV ile basit yüz değiştirme
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            print("✅ OpenCV yüz değiştirme sistemi hazır")
    
    def detect_and_align_face_insightface(self, image: np.ndarray) -> Optional[Dict]:
        """InsightFace ile yüz tespiti ve hizalama"""
        try:
            faces = self.face_analyzer.get(image)
            if len(faces) > 0:
                face = faces[0]  # İlk yüzü al
                return {
                    'face': face,
                    'bbox': face.bbox.astype(int),
                    'landmarks': face.kps if hasattr(face, 'kps') else None,
                    'embedding': face.embedding if hasattr(face, 'embedding') else None
                }
            return None
        except Exception as e:
            print(f"❌ InsightFace yüz tespiti hatası: {e}")
            return None
    
    def detect_face_opencv(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """OpenCV ile yüz tespiti"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # En büyük yüzü al
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                return tuple(largest_face)
            return None
        except Exception as e:
            print(f"❌ OpenCV yüz tespiti hatası: {e}")
            return None
    
    def swap_face_insightface(self, source_image: np.ndarray, target_image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """InsightFace ile yüz değiştirme"""
        try:
            # Kaynak yüzü tespit et
            source_face_data = self.detect_and_align_face_insightface(source_image)
            if source_face_data is None:
                return target_image, {'success': False, 'error': 'Kaynak görüntüde yüz bulunamadı'}
            
            # Hedef yüzleri tespit et
            target_faces = self.face_analyzer.get(target_image)
            if len(target_faces) == 0:
                return target_image, {'success': False, 'error': 'Hedef görüntüde yüz bulunamadı'}
            
            # Sonuç görüntüsü
            result_image = target_image.copy()
            swapped_count = 0
            
            # Her hedef yüzü değiştir
            for target_face in target_faces:
                try:
                    # Yüz değiştirme
                    result_image = self.swapper_model.get(
                        result_image, 
                        target_face, 
                        source_face_data['face'], 
                        paste_back=True
                    )
                    swapped_count += 1
                except Exception as e:
                    print(f"⚠️  Yüz değiştirme hatası: {e}")
                    continue
            
            return result_image, {
                'success': True,
                'swapped_faces': swapped_count,
                'total_target_faces': len(target_faces),
                'method': 'insightface'
            }
            
        except Exception as e:
            print(f"❌ InsightFace yüz değiştirme hatası: {e}")
            return target_image, {'success': False, 'error': str(e)}
    
    def swap_face_opencv(self, source_image: np.ndarray, target_image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """OpenCV ile basit yüz değiştirme"""
        try:
            # Kaynak yüzü tespit et
            source_face = self.detect_face_opencv(source_image)
            if source_face is None:
                return target_image, {'success': False, 'error': 'Kaynak görüntüde yüz bulunamadı'}
            
            # Hedef yüzü tespit et
            target_face = self.detect_face_opencv(target_image)
            if target_face is None:
                return target_image, {'success': False, 'error': 'Hedef görüntüde yüz bulunamadı'}
            
            # Yüzleri kırp
            sx, sy, sw, sh = source_face
            tx, ty, tw, th = target_face
            
            source_face_crop = source_image[sy:sy+sh, sx:sx+sw]
            target_face_crop = target_image[ty:ty+th, tx:tx+tw]
            
            # Yüzü yeniden boyutlandır
            resized_source = cv2.resize(source_face_crop, (tw, th))
            
            # Basit karıştırma (seamless cloning)
            mask = np.ones((th, tw), dtype=np.uint8) * 255
            
            # Merkez noktası
            center = (tx + tw // 2, ty + th // 2)
            
            # Seamless cloning uygula
            result_image = cv2.seamlessClone(
                resized_source, 
                target_image, 
                mask, 
                center, 
                cv2.NORMAL_CLONE
            )
            
            return result_image, {
                'success': True,
                'swapped_faces': 1,
                'total_target_faces': 1,
                'method': 'opencv'
            }
            
        except Exception as e:
            print(f"❌ OpenCV yüz değiştirme hatası: {e}")
            return target_image, {'success': False, 'error': str(e)}
    
    def swap_faces(self, source_path: str, target_path: str, output_path: Optional[str] = None) -> Dict:
        """Ana yüz değiştirme fonksiyonu"""
        # Dosya kontrolü
        if not os.path.exists(source_path):
            return {
                'success': False,
                'error': f'Kaynak dosya bulunamadı: {source_path}',
                'output_path': None
            }
        
        if not os.path.exists(target_path):
            return {
                'success': False,
                'error': f'Hedef dosya bulunamadı: {target_path}',
                'output_path': None
            }
        
        # Görüntüleri yükle
        source_image = cv2.imread(source_path)
        target_image = cv2.imread(target_path)
        
        if source_image is None:
            return {
                'success': False,
                'error': f'Kaynak görüntü yüklenemedi: {source_path}',
                'output_path': None
            }
        
        if target_image is None:
            return {
                'success': False,
                'error': f'Hedef görüntü yüklenemedi: {target_path}',
                'output_path': None
            }
        
        print(f"🔄 {self.method.upper()} ile yüz değiştirme yapılıyor...")
        print(f"   Kaynak: {source_path}")
        print(f"   Hedef: {target_path}")
        
        # Yönteme göre değiştirme yap
        if self.method == 'insightface':
            result_image, swap_info = self.swap_face_insightface(source_image, target_image)
        elif self.method == 'opencv':
            result_image, swap_info = self.swap_face_opencv(source_image, target_image)
        
        if not swap_info['success']:
            return {
                'success': False,
                'error': swap_info['error'],
                'output_path': None
            }
        
        # Çıktı dosya yolu belirle
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"swap_{self.method}_{timestamp}.jpg"
            output_path = f"outputs/swap/{filename}"
        
        # Çıktı klasörünü oluştur
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Sonuç görüntüyü kaydet
        success = cv2.imwrite(output_path, result_image)
        
        if success:
            print(f"✅ Yüz değiştirme tamamlandı!")
            print(f"   Değiştirilen yüz sayısı: {swap_info['swapped_faces']}")
            print(f"   Çıktı: {output_path}")
            
            return {
                'success': True,
                'source_path': source_path,
                'target_path': target_path,
                'output_path': output_path,
                'method': self.method,
                'swapped_faces': swap_info['swapped_faces'],
                'total_target_faces': swap_info['total_target_faces'],
                'original_size': target_image.shape[:2],
                'result_size': result_image.shape[:2],
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'success': False,
                'error': 'Sonuç görüntüsü kaydedilemedi',
                'output_path': None
            }
    
    def batch_swap(self, source_path: str, target_dir: str, output_dir: str = "outputs/swap_batch") -> Dict:
        """Tek kaynak yüzle birden fazla hedef görüntüde değiştirme"""
        if not os.path.exists(source_path):
            return {
                'success': False,
                'error': f'Kaynak dosya bulunamadı: {source_path}',
                'processed_count': 0
            }
        
        target_path = Path(target_dir)
        if not target_path.exists():
            return {
                'success': False,
                'error': f'Hedef klasör bulunamadı: {target_dir}',
                'processed_count': 0
            }
        
        # Desteklenen dosya formatları
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Hedef görüntü dosyalarını bul
        target_files = [f for f in target_path.iterdir() 
                       if f.suffix.lower() in supported_formats]
        
        if not target_files:
            return {
                'success': False,
                'error': f'Hedef klasörde görüntü dosyası bulunamadı: {target_dir}',
                'processed_count': 0
            }
        
        print(f"🔄 {len(target_files)} dosyada toplu yüz değiştirme...")
        
        # Çıktı klasörünü oluştur
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        success_count = 0
        
        for i, target_file in enumerate(target_files, 1):
            print(f"   [{i}/{len(target_files)}] {target_file.name}")
            
            # Çıktı dosya yolu
            output_file = output_path / f"swap_{target_file.name}"
            
            # Yüz değiştirme yap
            result = self.swap_faces(source_path, str(target_file), str(output_file))
            results.append(result)
            
            if result['success']:
                success_count += 1
        
        print(f"✅ Toplu yüz değiştirme tamamlandı: {success_count}/{len(target_files)} başarılı")
        
        return {
            'success': True,
            'source_path': source_path,
            'target_dir': target_dir,
            'output_dir': output_dir,
            'total_files': len(target_files),
            'processed_count': success_count,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def create_face_mosaic(self, image_path: str, output_path: Optional[str] = None) -> Dict:
        """Görüntüdeki tüm yüzleri mozaikle"""
        if not os.path.exists(image_path):
            return {
                'success': False,
                'error': f'Dosya bulunamadı: {image_path}',
                'output_path': None
            }
        
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'error': f'Görüntü yüklenemedi: {image_path}',
                'output_path': None
            }
        
        print(f"🔍 Yüz mozaikleme yapılıyor...")
        
        # Yüzleri tespit et
        if self.method == 'insightface':
            faces = self.face_analyzer.get(image)
            face_boxes = [face.bbox.astype(int) for face in faces]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            face_boxes = faces
        
        if len(face_boxes) == 0:
            return {
                'success': False,
                'error': 'Görüntüde yüz bulunamadı',
                'output_path': None
            }
        
        # Her yüzü mozaikle
        result_image = image.copy()
        for bbox in face_boxes:
            if self.method == 'insightface':
                x, y, x2, y2 = bbox
            else:
                x, y, w, h = bbox
                x2, y2 = x + w, y + h
            
            # Yüz bölgesini al
            face_region = result_image[y:y2, x:x2]
            
            # Mozaik uygula (küçült ve büyüt)
            small = cv2.resize(face_region, (10, 10))
            mosaic = cv2.resize(small, (x2-x, y2-y), interpolation=cv2.INTER_NEAREST)
            
            # Mozaiki yerleştir
            result_image[y:y2, x:x2] = mosaic
        
        # Çıktı dosya yolu belirle
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mosaic_{timestamp}.jpg"
            output_path = f"outputs/swap/{filename}"
        
        # Çıktı klasörünü oluştur
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Sonuç görüntüyü kaydet
        success = cv2.imwrite(output_path, result_image)
        
        if success:
            print(f"✅ Mozaikleme tamamlandı: {len(face_boxes)} yüz")
            print(f"   Çıktı: {output_path}")
            
            return {
                'success': True,
                'input_path': image_path,
                'output_path': output_path,
                'method': self.method,
                'mosaiced_faces': len(face_boxes),
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'success': False,
                'error': 'Mozaik görüntüsü kaydedilemedi',
                'output_path': None
            }

def main():
    parser = argparse.ArgumentParser(description='Gelişmiş Yüz Değiştirme Sistemi')
    parser.add_argument('--source', help='Kaynak yüz görüntüsü')
    parser.add_argument('--target', help='Hedef görüntü')
    parser.add_argument('--method', choices=['insightface', 'opencv'], 
                       default='insightface', help='Değiştirme yöntemi')
    parser.add_argument('--output', help='Çıktı dosya yolu')
    parser.add_argument('--batch', action='store_true', 
                       help='Toplu yüz değiştirme (target klasör olmalı)')
    parser.add_argument('--mosaic', help='Yüz mozaikleme (görüntü yolu)')
    
    args = parser.parse_args()
    
    # Yüz değiştirme sistemi oluştur
    swapper = FaceSwapper(method=args.method)
    
    if args.mosaic:
        # Yüz mozaikleme
        result = swapper.create_face_mosaic(args.mosaic, args.output)
        
        if result['success']:
            print(f"\n📊 Mozaikleme Sonuçları:")
            print(f"   Yöntem: {result['method']}")
            print(f"   Mozaiklenen yüz sayısı: {result['mosaiced_faces']}")
            print(f"   Çıktı: {result['output_path']}")
        else:
            print(f"❌ Hata: {result['error']}")
    
    elif args.batch and args.source and args.target:
        # Toplu yüz değiştirme
        result = swapper.batch_swap(args.source, args.target, args.output)
        
        if result['success']:
            print(f"\n📊 Toplu Değiştirme Sonuçları:")
            print(f"   Kaynak: {result['source_path']}")
            print(f"   Hedef klasör: {result['target_dir']}")
            print(f"   Toplam dosya: {result['total_files']}")
            print(f"   İşlenen: {result['processed_count']}")
            print(f"   Çıktı klasörü: {result['output_dir']}")
        else:
            print(f"❌ Hata: {result['error']}")
    
    elif args.source and args.target:
        # Tek dosya yüz değiştirme
        result = swapper.swap_faces(args.source, args.target, args.output)
        
        if result['success']:
            print(f"\n📊 Yüz Değiştirme Sonuçları:")
            print(f"   Yöntem: {result['method']}")
            print(f"   Değiştirilen yüz sayısı: {result['swapped_faces']}")
            print(f"   Toplam hedef yüz: {result['total_target_faces']}")
            print(f"   Orijinal boyut: {result['original_size']}")
            print(f"   Sonuç boyut: {result['result_size']}")
            print(f"   Çıktı: {result['output_path']}")
        else:
            print(f"❌ Hata: {result['error']}")
    
    else:
        print("❌ Lütfen gerekli parametreleri belirtin. --help ile yardım alın.")
        print("\nÖrnek kullanımlar:")
        print("  python face_swapper.py --source face1.jpg --target photo.jpg")
        print("  python face_swapper.py --source face1.jpg --target photos/ --batch")
        print("  python face_swapper.py --mosaic group_photo.jpg")

if __name__ == "__main__":
    main()
