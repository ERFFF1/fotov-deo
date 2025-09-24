#!/usr/bin/env python3
"""
AI Video Üretim Sistemi
Stable Video Diffusion ve diğer AI modelleri ile video üretimi
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import cv2
import json

class AIVideoGenerator:
    def __init__(self):
        """AI video üretici başlatıcı"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_models()
        
    def setup_models(self):
        """AI modellerini kur"""
        try:
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import load_image, export_to_video
            
            print(f"🎬 AI video modelleri kuruluyor... (Cihaz: {self.device})")
            
            # Stable Video Diffusion pipeline
            self.svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid",
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                variant="fp16" if self.device.type == 'cuda' else None
            ).to(self.device)
            
            print("✅ AI video modelleri hazır")
            
        except ImportError as e:
            print(f"⚠️  AI video modelleri kurulu değil: {e}")
            print("💡 Kurulum için: pip install diffusers transformers")
            self.svd_pipeline = None
    
    def generate_video_from_image(self, image_path: str, duration: int = 3, 
                                 fps: int = 24, motion_bucket_id: int = 127) -> Dict:
        """Görüntüden video üret"""
        if self.svd_pipeline is None:
            return {
                'success': False,
                'error': 'AI video modelleri kurulu değil',
                'output_path': None
            }
        
        try:
            # Görüntüyü yükle ve hazırla
            image = Image.open(image_path).convert('RGB')
            
            # Görüntüyü 1024x576 boyutuna getir (SVD için gerekli)
            image = image.resize((1024, 576))
            
            print(f"🎬 AI video üretiliyor...")
            print(f"   Kaynak görüntü: {image_path}")
            print(f"   Süre: {duration} saniye")
            print(f"   FPS: {fps}")
            
            # Video üret
            with torch.autocast(self.device.type):
                frames = self.svd_pipeline(
                    image,
                    decode_chunk_size=8,
                    motion_bucket_id=motion_bucket_id,
                    noise_aug_strength=0.02,
                    num_frames=duration * fps
                ).frames[0]
            
            # Çıktı dosya yolu
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_video_{timestamp}.mp4"
            output_path = f"outputs/ai_videos/{filename}"
            
            # Çıktı klasörünü oluştur
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Video'yu kaydet
            export_to_video(frames, output_path, fps=fps)
            
            print(f"✅ AI video üretildi: {output_path}")
            
            return {
                'success': True,
                'output_path': output_path,
                'source_image': image_path,
                'duration': duration,
                'fps': fps,
                'motion_bucket_id': motion_bucket_id,
                'num_frames': len(frames),
                'generation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'AI video üretim hatası: {e}',
                'output_path': None
            }
    
    def generate_video_from_prompt(self, prompt: str, duration: int = 3, 
                                  fps: int = 24, size: str = '512x512') -> Dict:
        """Prompt'tan video üret (görüntü + video kombinasyonu)"""
        try:
            # Önce görüntü üret
            from ai_photo_generator import AIPhotoGenerator
            photo_gen = AIPhotoGenerator()
            
            if photo_gen.sd_pipeline is None:
                return {
                    'success': False,
                    'error': 'AI fotoğraf modelleri kurulu değil',
                    'output_path': None
                }
            
            print(f"🎨 Önce görüntü üretiliyor...")
            
            # Görüntü üret
            image_result = photo_gen.generate_image(
                prompt=prompt,
                style='realistic',
                size=size,
                steps=20
            )
            
            if not image_result['success']:
                return {
                    'success': False,
                    'error': f'Görüntü üretim hatası: {image_result["error"]}',
                    'output_path': None
                }
            
            # Görüntüden video üret
            video_result = self.generate_video_from_image(
                image_path=image_result['output_path'],
                duration=duration,
                fps=fps
            )
            
            if video_result['success']:
                # Geçici görüntü dosyasını sil
                try:
                    os.remove(image_result['output_path'])
                except:
                    pass
                
                return {
                    'success': True,
                    'output_path': video_result['output_path'],
                    'prompt': prompt,
                    'duration': duration,
                    'fps': fps,
                    'size': size,
                    'generation_time': datetime.now().isoformat()
                }
            else:
                return video_result
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Prompt video üretim hatası: {e}',
                'output_path': None
            }
    
    def create_slideshow_video(self, image_paths: List[str], duration_per_image: float = 2.0, 
                              fps: int = 24, transition_type: str = 'fade') -> Dict:
        """Görüntü koleksiyonundan slideshow video oluştur"""
        try:
            if not image_paths:
                return {
                    'success': False,
                    'error': 'Görüntü listesi boş',
                    'output_path': None
                }
            
            print(f"🎬 Slideshow video oluşturuluyor...")
            print(f"   Görüntü sayısı: {len(image_paths)}")
            print(f"   Görüntü başına süre: {duration_per_image} saniye")
            
            # Video writer oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"slideshow_{timestamp}.mp4"
            output_path = f"outputs/ai_videos/{filename}"
            
            # Çıktı klasörünü oluştur
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # İlk görüntüden boyutları al
            first_image = cv2.imread(image_paths[0])
            height, width, _ = first_image.shape
            
            # Video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frames_per_image = int(duration_per_image * fps)
            
            for i, image_path in enumerate(image_paths):
                if not os.path.exists(image_path):
                    print(f"⚠️  Görüntü bulunamadı: {image_path}")
                    continue
                
                # Görüntüyü yükle ve boyutlandır
                image = cv2.imread(image_path)
                image = cv2.resize(image, (width, height))
                
                # Geçiş efekti ekle
                if transition_type == 'fade' and i > 0:
                    # Fade in efekti
                    for alpha in np.linspace(0, 1, frames_per_image // 4):
                        blended = cv2.addWeighted(image, alpha, image, 1-alpha, 0)
                        out.write(blended)
                    
                    # Ana görüntü
                    for _ in range(frames_per_image - frames_per_image // 2):
                        out.write(image)
                    
                    # Fade out efekti
                    for alpha in np.linspace(1, 0, frames_per_image // 4):
                        blended = cv2.addWeighted(image, alpha, image, 1-alpha, 0)
                        out.write(blended)
                else:
                    # Basit geçiş
                    for _ in range(frames_per_image):
                        out.write(image)
                
                print(f"   Görüntü {i+1}/{len(image_paths)} işlendi")
            
            out.release()
            
            print(f"✅ Slideshow video oluşturuldu: {output_path}")
            
            return {
                'success': True,
                'output_path': output_path,
                'image_paths': image_paths,
                'duration_per_image': duration_per_image,
                'fps': fps,
                'transition_type': transition_type,
                'total_duration': len(image_paths) * duration_per_image,
                'generation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Slideshow video oluşturma hatası: {e}',
                'output_path': None
            }
    
    def add_audio_to_video(self, video_path: str, audio_path: str, output_path: str = None) -> Dict:
        """Videoya ses ekle"""
        try:
            import subprocess
            
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
            
            # Çıktı dosya yolu belirle
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"video_with_audio_{timestamp}.mp4"
                output_path = f"outputs/ai_videos/{filename}"
            
            # Çıktı klasörünü oluştur
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            print(f"🔊 Videoya ses ekleniyor...")
            
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
                'output_path': output_path,
                'video_path': video_path,
                'audio_path': audio_path,
                'generation_time': datetime.now().isoformat()
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
        except Exception as e:
            return {
                'success': False,
                'error': f'Ses ekleme hatası: {e}',
                'output_path': None
            }
    
    def create_animated_text_video(self, text: str, duration: int = 5, 
                                  fps: int = 24, size: str = '1920x1080') -> Dict:
        """Animasyonlu metin video oluştur"""
        try:
            width, height = map(int, size.split('x'))
            
            print(f"🎬 Animasyonlu metin video oluşturuluyor...")
            print(f"   Metin: {text}")
            print(f"   Boyut: {width}x{height}")
            
            # Video writer
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"text_animation_{timestamp}.mp4"
            output_path = f"outputs/ai_videos/{filename}"
            
            # Çıktı klasörünü oluştur
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            total_frames = duration * fps
            
            for frame_num in range(total_frames):
                # Boş frame oluştur
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame.fill(0)  # Siyah arka plan
                
                # Metin animasyonu
                progress = frame_num / total_frames
                
                # Fade in efekti
                alpha = min(1.0, progress * 3)  # İlk 1/3'te fade in
                
                # Metin rengi (beyaz)
                color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
                
                # Metin boyutu
                font_scale = 2.0
                thickness = 3
                
                # Metni ortala
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (width - text_size[0]) // 2
                text_y = (height + text_size[1]) // 2
                
                # Metni çiz
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
                
                # Frame'i video'ya ekle
                out.write(frame)
            
            out.release()
            
            print(f"✅ Animasyonlu metin video oluşturuldu: {output_path}")
            
            return {
                'success': True,
                'output_path': output_path,
                'text': text,
                'duration': duration,
                'fps': fps,
                'size': size,
                'generation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Metin animasyon hatası: {e}',
                'output_path': None
            }
    
    def get_video_info(self, video_path: str) -> Dict:
        """Video bilgilerini al"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {
                    'success': False,
                    'error': 'Video açılamadı',
                    'info': None
                }
            
            # Video özellikleri
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'success': True,
                'info': {
                    'fps': fps,
                    'frame_count': frame_count,
                    'width': width,
                    'height': height,
                    'duration': duration,
                    'file_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Video bilgi alma hatası: {e}',
                'info': None
            }

def main():
    """Test fonksiyonu"""
    generator = AIVideoGenerator()
    
    if generator.svd_pipeline is None:
        print("❌ AI video modelleri kurulu değil!")
        print("💡 Kurulum için: pip install diffusers transformers")
        return
    
    # Test video üretimi
    result = generator.generate_video_from_prompt(
        prompt="a beautiful sunset over the ocean",
        duration=3,
        fps=24
    )
    
    if result['success']:
        print(f"✅ Test video üretildi: {result['output_path']}")
    else:
        print(f"❌ Test hatası: {result['error']}")

if __name__ == "__main__":
    main()
