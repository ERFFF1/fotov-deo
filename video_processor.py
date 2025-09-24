#!/usr/bin/env python3
"""
Gelişmiş Video İşleme ve Yüz Yerleştirme Sistemi
FFmpeg ve OpenCV kullanarak video üzerinde yüz işlemleri yapar
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
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, method='insightface'):
        """
        Video işleme sistemi başlatıcı
        
        Args:
            method (str): 'insightface', 'opencv' seçenekleri
        """
        self.method = method
        self.setup_processor()
        
    def setup_processor(self):
        """Seçilen yönteme göre işleme sistemini kur"""
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
                
                print("✅ InsightFace video işleme sistemi hazır")
                
            except ImportError:
                print("❌ InsightFace kurulu değil, OpenCV'ye geçiliyor")
                self.method = 'opencv'
                self.setup_processor()
                
        elif self.method == 'opencv':
            # OpenCV ile basit video işleme
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            print("✅ OpenCV video işleme sistemi hazır")
    
    def extract_frames(self, video_path: str, output_dir: str, fps: Optional[float] = None) -> Dict:
        """Videodan frame'leri çıkar"""
        if not os.path.exists(video_path):
            return {
                'success': False,
                'error': f'Video dosyası bulunamadı: {video_path}',
                'frame_count': 0
            }
        
        # Çıktı klasörünü oluştur
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"🎬 Frame'ler çıkarılıyor: {video_path}")
        
        try:
            # Video bilgilerini al
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            
            if fps is None:
                fps = original_fps
            
            frame_interval = int(original_fps / fps) if fps < original_fps else 1
            
            print(f"   Toplam frame: {total_frames}")
            print(f"   Orijinal FPS: {original_fps:.2f}")
            print(f"   Hedef FPS: {fps:.2f}")
            print(f"   Frame aralığı: {frame_interval}")
            
            frame_count = 0
            saved_count = 0
            
            with tqdm(total=total_frames, desc="Frame çıkarılıyor") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Belirtilen aralıkta frame'leri kaydet
                    if frame_count % frame_interval == 0:
                        frame_filename = f"frame_{saved_count:06d}.jpg"
                        frame_path = Path(output_dir) / frame_filename
                        cv2.imwrite(str(frame_path), frame)
                        saved_count += 1
                    
                    frame_count += 1
                    pbar.update(1)
            
            cap.release()
            
            print(f"✅ {saved_count} frame çıkarıldı: {output_dir}")
            
            return {
                'success': True,
                'video_path': video_path,
                'output_dir': output_dir,
                'total_frames': total_frames,
                'extracted_frames': saved_count,
                'original_fps': original_fps,
                'target_fps': fps,
                'frame_interval': frame_interval
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Frame çıkarma hatası: {e}',
                'frame_count': 0
            }
    
    def process_frame_insightface(self, frame: np.ndarray, source_face_data: Dict) -> np.ndarray:
        """InsightFace ile frame işleme"""
        try:
            # Hedef yüzleri tespit et
            target_faces = self.face_analyzer.get(frame)
            if len(target_faces) == 0:
                return frame
            
            # Her hedef yüzü değiştir
            result_frame = frame.copy()
            for target_face in target_faces:
                try:
                    result_frame = self.swapper_model.get(
                        result_frame, 
                        target_face, 
                        source_face_data['face'], 
                        paste_back=True
                    )
                except Exception as e:
                    print(f"⚠️  Frame işleme hatası: {e}")
                    continue
            
            return result_frame
            
        except Exception as e:
            print(f"❌ InsightFace frame işleme hatası: {e}")
            return frame
    
    def process_frame_opencv(self, frame: np.ndarray, source_face_crop: np.ndarray) -> np.ndarray:
        """OpenCV ile frame işleme"""
        try:
            # Hedef yüzü tespit et
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return frame
            
            # En büyük yüzü al
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Kaynak yüzü yeniden boyutlandır
            resized_source = cv2.resize(source_face_crop, (w, h))
            
            # Basit karıştırma
            mask = np.ones((h, w), dtype=np.uint8) * 255
            center = (x + w // 2, y + h // 2)
            
            result_frame = cv2.seamlessClone(
                resized_source, 
                frame, 
                mask, 
                center, 
                cv2.NORMAL_CLONE
            )
            
            return result_frame
            
        except Exception as e:
            print(f"❌ OpenCV frame işleme hatası: {e}")
            return frame
    
    def process_frames(self, frames_dir: str, source_path: str, output_dir: str) -> Dict:
        """Frame'leri toplu işle"""
        if not os.path.exists(frames_dir):
            return {
                'success': False,
                'error': f'Frame klasörü bulunamadı: {frames_dir}',
                'processed_count': 0
            }
        
        if not os.path.exists(source_path):
            return {
                'success': False,
                'error': f'Kaynak dosya bulunamadı: {source_path}',
                'processed_count': 0
            }
        
        # Çıktı klasörünü oluştur
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Kaynak yüzü hazırla
        source_image = cv2.imread(source_path)
        if source_image is None:
            return {
                'success': False,
                'error': f'Kaynak görüntü yüklenemedi: {source_path}',
                'processed_count': 0
            }
        
        if self.method == 'insightface':
            source_face_data = self.detect_and_align_face_insightface(source_image)
            if source_face_data is None:
                return {
                    'success': False,
                    'error': 'Kaynak görüntüde yüz bulunamadı',
                    'processed_count': 0
                }
        else:
            source_face = self.detect_face_opencv(source_image)
            if source_face is None:
                return {
                    'success': False,
                    'error': 'Kaynak görüntüde yüz bulunamadı',
                    'processed_count': 0
                }
            sx, sy, sw, sh = source_face
            source_face_crop = source_image[sy:sy+sh, sx:sx+sw]
        
        # Frame dosyalarını bul
        frame_files = sorted([f for f in Path(frames_dir).iterdir() 
                            if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        
        if not frame_files:
            return {
                'success': False,
                'error': f'Frame dosyası bulunamadı: {frames_dir}',
                'processed_count': 0
            }
        
        print(f"🔄 {len(frame_files)} frame işleniyor...")
        
        processed_count = 0
        
        with tqdm(total=len(frame_files), desc="Frame işleniyor") as pbar:
            for frame_file in frame_files:
                try:
                    # Frame'i yükle
                    frame = cv2.imread(str(frame_file))
                    if frame is None:
                        continue
                    
                    # Frame'i işle
                    if self.method == 'insightface':
                        processed_frame = self.process_frame_insightface(frame, source_face_data)
                    else:
                        processed_frame = self.process_frame_opencv(frame, source_face_crop)
                    
                    # İşlenmiş frame'i kaydet
                    output_file = Path(output_dir) / frame_file.name
                    cv2.imwrite(str(output_file), processed_frame)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"⚠️  Frame işleme hatası {frame_file.name}: {e}")
                    continue
                
                pbar.update(1)
        
        print(f"✅ {processed_count}/{len(frame_files)} frame işlendi")
        
        return {
            'success': True,
            'frames_dir': frames_dir,
            'source_path': source_path,
            'output_dir': output_dir,
            'total_frames': len(frame_files),
            'processed_count': processed_count,
            'method': self.method
        }
    
    def create_video_from_frames(self, frames_dir: str, output_path: str, fps: float = 25.0) -> Dict:
        """Frame'lerden video oluştur"""
        if not os.path.exists(frames_dir):
            return {
                'success': False,
                'error': f'Frame klasörü bulunamadı: {frames_dir}',
                'output_path': None
            }
        
        # Frame dosyalarını bul
        frame_files = sorted([f for f in Path(frames_dir).iterdir() 
                            if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        
        if not frame_files:
            return {
                'success': False,
                'error': f'Frame dosyası bulunamadı: {frames_dir}',
                'output_path': None
            }
        
        print(f"🎬 Video oluşturuluyor: {len(frame_files)} frame, {fps} FPS")
        
        try:
            # İlk frame'den video özelliklerini al
            first_frame = cv2.imread(str(frame_files[0]))
            height, width, _ = first_frame.shape
            
            # Video writer oluştur
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Frame'leri video'ya ekle
            with tqdm(total=len(frame_files), desc="Video oluşturuluyor") as pbar:
                for frame_file in frame_files:
                    frame = cv2.imread(str(frame_file))
                    if frame is not None:
                        out.write(frame)
                    pbar.update(1)
            
            out.release()
            
            print(f"✅ Video oluşturuldu: {output_path}")
            
            return {
                'success': True,
                'frames_dir': frames_dir,
                'output_path': output_path,
                'frame_count': len(frame_files),
                'fps': fps,
                'resolution': (width, height)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Video oluşturma hatası: {e}',
                'output_path': None
            }
    
    def swap_face_in_video(self, video_path: str, source_face_path: str, output_path: Optional[str] = None) -> Dict:
        """Videoda yüz değiştirme (ana fonksiyon)"""
        if not os.path.exists(video_path):
            return {
                'success': False,
                'error': f'Video dosyası bulunamadı: {video_path}',
                'output_path': None
            }
        
        if not os.path.exists(source_face_path):
            return {
                'success': False,
                'error': f'Kaynak yüz dosyası bulunamadı: {source_face_path}',
                'output_path': None
            }
        
        # Çıktı dosya yolu belirle
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_swap_{timestamp}.mp4"
            output_path = f"outputs/videos/{filename}"
        
        # Geçici klasörler
        with tempfile.TemporaryDirectory() as temp_dir:
            frames_dir = Path(temp_dir) / "frames"
            processed_dir = Path(temp_dir) / "processed"
            
            print(f"🎬 Video yüz değiştirme başlıyor...")
            print(f"   Video: {video_path}")
            print(f"   Kaynak yüz: {source_face_path}")
            print(f"   Çıktı: {output_path}")
            
            # 1. Frame'leri çıkar
            extract_result = self.extract_frames(video_path, str(frames_dir))
            if not extract_result['success']:
                return extract_result
            
            # 2. Frame'leri işle
            process_result = self.process_frames(str(frames_dir), source_face_path, str(processed_dir))
            if not process_result['success']:
                return process_result
            
            # 3. Video oluştur
            create_result = self.create_video_from_frames(str(processed_dir), output_path, extract_result['original_fps'])
            if not create_result['success']:
                return create_result
            
            # Çıktı klasörünü oluştur
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            print(f"✅ Video yüz değiştirme tamamlandı!")
            print(f"   İşlenen frame: {process_result['processed_count']}")
            print(f"   Çıktı: {output_path}")
            
            return {
                'success': True,
                'video_path': video_path,
                'source_face_path': source_face_path,
                'output_path': output_path,
                'method': self.method,
                'total_frames': extract_result['total_frames'],
                'processed_frames': process_result['processed_count'],
                'fps': create_result['fps'],
                'resolution': create_result['resolution'],
                'timestamp': datetime.now().isoformat()
            }
    
    def add_audio_to_video(self, video_path: str, audio_path: str, output_path: str) -> Dict:
        """Videoya ses ekle"""
        if not os.path.exists(video_path):
            return {
                'success': False,
                'error': f'Video dosyası bulunamadı: {video_path}',
                'output_path': None
            }
        
        if not os.path.exists(audio_path):
            return {
                'success': False,
                'error': f'Ses dosyası bulunamadı: {audio_path}',
                'output_path': None
            }
        
        print(f"🔊 Videoya ses ekleniyor...")
        
        try:
            # FFmpeg ile ses ekleme
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', video_path,  # Video input
                '-i', audio_path,  # Audio input
                '-c:v', 'copy',  # Copy video codec
                '-c:a', 'aac',  # Audio codec
                '-map', '0:v:0',  # Map video from first input
                '-map', '1:a:0',  # Map audio from second input
                '-shortest',  # End when shortest input ends
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print(f"✅ Ses eklendi: {output_path}")
            
            return {
                'success': True,
                'video_path': video_path,
                'audio_path': audio_path,
                'output_path': output_path,
                'timestamp': datetime.now().isoformat()
            }
            
        except subprocess.CalledProcessError as e:
            return {
                'success': False,
                'error': f'Ses ekleme hatası: {e.stderr}',
                'output_path': None
            }
        except FileNotFoundError:
            return {
                'success': False,
                'error': 'FFmpeg bulunamadı. Lütfen FFmpeg kurun.',
                'output_path': None
            }
    
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

def main():
    parser = argparse.ArgumentParser(description='Gelişmiş Video İşleme Sistemi')
    parser.add_argument('--video', help='İşlenecek video dosyası')
    parser.add_argument('--source-face', help='Kaynak yüz görüntüsü')
    parser.add_argument('--method', choices=['insightface', 'opencv'], 
                       default='insightface', help='İşleme yöntemi')
    parser.add_argument('--output', help='Çıktı video yolu')
    parser.add_argument('--audio', help='Eklenecek ses dosyası')
    parser.add_argument('--extract-frames', help='Frame çıkarma (video yolu)')
    parser.add_argument('--frames-dir', help='Frame klasörü')
    parser.add_argument('--fps', type=float, help='Hedef FPS')
    
    args = parser.parse_args()
    
    # Video işleme sistemi oluştur
    processor = VideoProcessor(method=args.method)
    
    if args.extract_frames:
        # Frame çıkarma
        output_dir = args.frames_dir or "work/frames"
        result = processor.extract_frames(args.extract_frames, output_dir, args.fps)
        
        if result['success']:
            print(f"\n📊 Frame Çıkarma Sonuçları:")
            print(f"   Video: {result['video_path']}")
            print(f"   Çıktı klasörü: {result['output_dir']}")
            print(f"   Toplam frame: {result['total_frames']}")
            print(f"   Çıkarılan frame: {result['extracted_frames']}")
            print(f"   FPS: {result['original_fps']:.2f} -> {result['target_fps']:.2f}")
        else:
            print(f"❌ Hata: {result['error']}")
    
    elif args.video and args.source_face:
        # Video yüz değiştirme
        result = processor.swap_face_in_video(args.video, args.source_face, args.output)
        
        if result['success']:
            print(f"\n📊 Video Yüz Değiştirme Sonuçları:")
            print(f"   Yöntem: {result['method']}")
            print(f"   Toplam frame: {result['total_frames']}")
            print(f"   İşlenen frame: {result['processed_frames']}")
            print(f"   FPS: {result['fps']}")
            print(f"   Çözünürlük: {result['resolution']}")
            print(f"   Çıktı: {result['output_path']}")
            
            # Ses ekleme
            if args.audio:
                audio_result = processor.add_audio_to_video(
                    result['output_path'], 
                    args.audio, 
                    result['output_path'].replace('.mp4', '_with_audio.mp4')
                )
                
                if audio_result['success']:
                    print(f"🔊 Ses eklendi: {audio_result['output_path']}")
                else:
                    print(f"❌ Ses ekleme hatası: {audio_result['error']}")
        else:
            print(f"❌ Hata: {result['error']}")
    
    else:
        print("❌ Lütfen gerekli parametreleri belirtin. --help ile yardım alın.")
        print("\nÖrnek kullanımlar:")
        print("  python video_processor.py --video input.mp4 --source-face face.jpg")
        print("  python video_processor.py --extract-frames input.mp4 --frames-dir frames/")
        print("  python video_processor.py --video input.mp4 --source-face face.jpg --audio audio.wav")

if __name__ == "__main__":
    main()
