#!/usr/bin/env python3
"""
Advanced AI Pipeline
ControlNet, gelişmiş AI özellikleri ve video post-prodüksiyon
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime

# Proje kök dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import JobConfig
from core.device_manager import get_device_manager, get_model_cache
from core.ethics_security import get_watermark_manager

class AdvancedAIPipeline:
    """Gelişmiş AI işleme pipeline'ı"""
    
    def __init__(self):
        self.setup_directories()
        self.device_manager = get_device_manager()
        self.model_cache = get_model_cache()
        self.watermark_manager = get_watermark_manager()
        
    def setup_directories(self):
        """Gerekli klasörleri oluştur"""
        directories = [
            "storage/outputs/advanced_ai",
            "storage/artifacts/advanced_ai",
            "storage/cache/advanced_ai"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def execute(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Pipeline'ı çalıştır"""
        job_type = config.job_type
        
        if job_type == 'controlnet_generation':
            return self._controlnet_generation(config, progress_callback)
        elif job_type == 'video_post_production':
            return self._video_post_production(config, progress_callback)
        elif job_type == 'advanced_talking_head':
            return self._advanced_talking_head(config, progress_callback)
        elif job_type == 'style_transfer':
            return self._style_transfer(config, progress_callback)
        else:
            return {'success': False, 'error': f'Desteklenmeyen gelişmiş AI işlemi: {job_type}'}
    
    def _controlnet_generation(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """ControlNet ile kontrollü görüntü üretimi"""
        try:
            progress_callback('ControlNet başlatılıyor', 10)
            
            # Parametreleri al
            control_image_path = config.inputs.get('control_image')
            prompt = config.params.get('prompt', '')
            control_type = config.params.get('control_type', 'canny')
            
            if not control_image_path or not os.path.exists(control_image_path):
                return {'success': False, 'error': 'Kontrol görüntüsü bulunamadı'}
            
            if not prompt:
                return {'success': False, 'error': 'Prompt gerekli'}
            
            progress_callback('ControlNet modeli yükleniyor', 30)
            
            # AI fotoğraf üretici
            from ai_photo_generator import AIPhotoGenerator
            
            # En uygun cihazı seç
            device = self.device_manager.get_optimal_device('ai_photo_generation', 4.0)
            self.device_manager.allocate_device(device)
            
            try:
                photo_gen = AIPhotoGenerator()
                
                progress_callback('ControlNet ile görüntü üretiliyor', 60)
                
                # ControlNet ile üretim
                result = photo_gen.generate_with_controlnet(
                    prompt=prompt,
                    control_image_path=control_image_path,
                    control_type=control_type,
                    steps=config.params.get('steps', 20),
                    guidance_scale=config.params.get('guidance_scale', 7.5),
                    controlnet_scale=config.params.get('controlnet_scale', 1.0)
                )
                
                if not result['success']:
                    return result
                
                # Filigran ekle
                progress_callback('Filigran ekleniyor', 90)
                
                watermarked_path = self.watermark_manager.add_watermark(
                    result['output_path'],
                    config.consent_tag
                )
                
                progress_callback('ControlNet üretimi tamamlandı', 100)
                
                return {
                    'success': True,
                    'output_path': watermarked_path,
                    'original_path': result['output_path'],
                    'metrics': {
                        'processing_time': datetime.now().isoformat(),
                        'control_type': control_type,
                        'device_used': device,
                        'prompt': prompt
                    }
                }
                
            finally:
                self.device_manager.release_device(device)
            
        except Exception as e:
            return {'success': False, 'error': f'ControlNet hatası: {str(e)}'}
    
    def _video_post_production(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Video post-prodüksiyon işlemi"""
        try:
            progress_callback('Video post-prodüksiyon başlatılıyor', 10)
            
            # Parametreleri al
            video_path = config.inputs.get('video')
            
            if not video_path or not os.path.exists(video_path):
                return {'success': False, 'error': 'Video dosyası bulunamadı'}
            
            progress_callback('Video analiz ediliyor', 20)
            
            # Video bilgilerini al
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Post-prodüksiyon adımları
            steps = []
            
            # 1. Renk düzeltme
            if config.params.get('color_correction', True):
                steps.append(('color_correction', 'Renk düzeltme'))
            
            # 2. Stabilizasyon
            if config.params.get('stabilization', False):
                steps.append(('stabilization', 'Stabilizasyon'))
            
            # 3. Gürültü azaltma
            if config.params.get('denoising', False):
                steps.append(('denoising', 'Gürültü azaltma'))
            
            # 4. Keskinlik artırma
            if config.params.get('sharpening', False):
                steps.append(('sharpening', 'Keskinlik artırma'))
            
            # 5. Ses iyileştirme
            if config.params.get('audio_enhancement', True):
                steps.append(('audio_enhancement', 'Ses iyileştirme'))
            
            progress_callback(f'{len(steps)} post-prodüksiyon adımı uygulanacak', 30)
            
            # Geçici dosya
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_output = f"storage/outputs/advanced_ai/post_prod_{timestamp}.mp4"
            Path(temp_output).parent.mkdir(parents=True, exist_ok=True)
            
            current_video = video_path
            
            # Her adımı uygula
            for i, (step_type, step_name) in enumerate(steps):
                progress_callback(f'{step_name} uygulanıyor', 30 + (i * 60 / len(steps)))
                
                if step_type == 'color_correction':
                    current_video = self._apply_color_correction(current_video, config.params)
                elif step_type == 'stabilization':
                    current_video = self._apply_stabilization(current_video)
                elif step_type == 'denoising':
                    current_video = self._apply_denoising(current_video)
                elif step_type == 'sharpening':
                    current_video = self._apply_sharpening(current_video)
                elif step_type == 'audio_enhancement':
                    current_video = self._apply_audio_enhancement(current_video, config.params)
            
            # Final çıktı
            final_output = f"storage/outputs/advanced_ai/post_prod_final_{timestamp}.mp4"
            
            if current_video != video_path:
                # Geçici dosyayı final konuma taşı
                import shutil
                shutil.move(current_video, final_output)
            else:
                # Hiçbir işlem yapılmadıysa orijinal dosyayı kopyala
                import shutil
                shutil.copy2(video_path, final_output)
            
            progress_callback('Video post-prodüksiyon tamamlandı', 100)
            
            return {
                'success': True,
                'output_path': final_output,
                'metrics': {
                    'processing_time': datetime.now().isoformat(),
                    'steps_applied': len(steps),
                    'original_fps': fps,
                    'original_frames': frame_count,
                    'resolution': f'{width}x{height}'
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Video post-prodüksiyon hatası: {str(e)}'}
    
    def _advanced_talking_head(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Gelişmiş konuşan kafa üretimi"""
        try:
            progress_callback('Gelişmiş konuşan kafa başlatılıyor', 10)
            
            # Parametreleri al
            image_path = config.inputs.get('image')
            audio_path = config.inputs.get('audio')
            
            if not image_path or not os.path.exists(image_path):
                return {'success': False, 'error': 'Görüntü dosyası bulunamadı'}
            
            if not audio_path or not os.path.exists(audio_path):
                return {'success': False, 'error': 'Ses dosyası bulunamadı'}
            
            progress_callback('Konuşan kafa üretiliyor', 30)
            
            # Temel konuşan kafa üretimi
            from talking_head import TalkingHeadGenerator
            
            method = config.params.get('method', 'wav2lip')
            generator = TalkingHeadGenerator(method=method)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_output = f"storage/outputs/advanced_ai/talking_temp_{timestamp}.mp4"
            Path(temp_output).parent.mkdir(parents=True, exist_ok=True)
            
            result = generator.generate_talking_head(image_path, audio_path, temp_output)
            
            if not result['success']:
                return result
            
            progress_callback('Post-prodüksiyon uygulanıyor', 70)
            
            # Post-prodüksiyon adımları
            final_output = f"storage/outputs/advanced_ai/talking_final_{timestamp}.mp4"
            
            # Stabilizasyon
            if config.params.get('stabilize', True):
                stabilized = self._stabilize_video(temp_output)
                if stabilized:
                    temp_output = stabilized
            
            # Keskinlik artırma
            if config.params.get('sharpen', True):
                sharpened = self._sharpen_video(temp_output)
                if sharpened:
                    temp_output = sharpened
            
            # Grain ekleme
            if config.params.get('add_grain', False):
                grained = self._add_grain(temp_output)
                if grained:
                    temp_output = grained
            
            # Final dosyayı oluştur
            import shutil
            shutil.move(temp_output, final_output)
            
            progress_callback('Gelişmiş konuşan kafa tamamlandı', 100)
            
            return {
                'success': True,
                'output_path': final_output,
                'metrics': {
                    'processing_time': datetime.now().isoformat(),
                    'method': method,
                    'post_processing': {
                        'stabilized': config.params.get('stabilize', True),
                        'sharpened': config.params.get('sharpen', True),
                        'grain_added': config.params.get('add_grain', False)
                    }
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Gelişmiş konuşan kafa hatası: {str(e)}'}
    
    def _style_transfer(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Stil transferi"""
        try:
            progress_callback('Stil transferi başlatılıyor', 10)
            
            # Parametreleri al
            content_image = config.inputs.get('content_image')
            style_image = config.inputs.get('style_image')
            
            if not content_image or not os.path.exists(content_image):
                return {'success': False, 'error': 'İçerik görüntüsü bulunamadı'}
            
            if not style_image or not os.path.exists(style_image):
                return {'success': False, 'error': 'Stil görüntüsü bulunamadı'}
            
            progress_callback('Stil transferi uygulanıyor', 50)
            
            # Basit stil transferi (OpenCV ile)
            content = cv2.imread(content_image)
            style = cv2.imread(style_image)
            
            # Boyutları eşitle
            style = cv2.resize(style, (content.shape[1], content.shape[0]))
            
            # Basit karıştırma
            alpha = config.params.get('style_strength', 0.5)
            result = cv2.addWeighted(content, 1-alpha, style, alpha, 0)
            
            # Çıktı dosyası
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"storage/outputs/advanced_ai/style_transfer_{timestamp}.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(output_path, result)
            
            progress_callback('Stil transferi tamamlandı', 100)
            
            return {
                'success': True,
                'output_path': output_path,
                'metrics': {
                    'processing_time': datetime.now().isoformat(),
                    'style_strength': alpha
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Stil transferi hatası: {str(e)}'}
    
    # Yardımcı fonksiyonlar
    def _apply_color_correction(self, video_path: str, params: Dict[str, Any]) -> str:
        """Renk düzeltme uygula"""
        # FFmpeg ile renk düzeltme
        import subprocess
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"storage/outputs/advanced_ai/color_corrected_{timestamp}.mp4"
        
        saturation = params.get('saturation', 1.0)
        contrast = params.get('contrast', 1.0)
        brightness = params.get('brightness', 0.0)
        
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vf', f'eq=saturation={saturation}:contrast={contrast}:brightness={brightness}',
            '-c:a', 'copy',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError:
            return video_path  # Hata durumunda orijinal dosyayı döndür
    
    def _apply_stabilization(self, video_path: str) -> str:
        """Video stabilizasyonu uygula"""
        import subprocess
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"storage/outputs/advanced_ai/stabilized_{timestamp}.mp4"
        
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vf', 'vidstabdetect=stepsize=6:shakiness=8:accuracy=9:result=transforms.trf',
            '-f', 'null', '-'
        ]
        
        try:
            # Transform dosyası oluştur
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Stabilizasyon uygula
            cmd2 = [
                'ffmpeg', '-y', '-i', video_path,
                '-vf', 'vidstabtransform=input=transforms.trf:smoothing=30',
                '-c:a', 'copy',
                output_path
            ]
            
            subprocess.run(cmd2, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError:
            return video_path
    
    def _apply_denoising(self, video_path: str) -> str:
        """Gürültü azaltma uygula"""
        import subprocess
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"storage/outputs/advanced_ai/denoised_{timestamp}.mp4"
        
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vf', 'hqdn3d=4:3:6:4.5',
            '-c:a', 'copy',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError:
            return video_path
    
    def _apply_sharpening(self, video_path: str) -> str:
        """Keskinlik artırma uygula"""
        import subprocess
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"storage/outputs/advanced_ai/sharpened_{timestamp}.mp4"
        
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vf', 'unsharp=5:5:0.8:3:3:0.4',
            '-c:a', 'copy',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError:
            return video_path
    
    def _apply_audio_enhancement(self, video_path: str, params: Dict[str, Any]) -> str:
        """Ses iyileştirme uygula"""
        import subprocess
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"storage/outputs/advanced_ai/audio_enhanced_{timestamp}.mp4"
        
        # Ses filtreleri
        audio_filters = []
        
        if params.get('normalize_audio', True):
            audio_filters.append('loudnorm=I=-16:TP=-1.5:LRA=11')
        
        if params.get('remove_noise', False):
            audio_filters.append('afftdn=nf=-25')
        
        cmd = ['ffmpeg', '-y', '-i', video_path]
        
        if audio_filters:
            cmd.extend(['-af', ','.join(audio_filters)])
        
        cmd.extend(['-c:v', 'copy', output_path])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError:
            return video_path
    
    def _stabilize_video(self, video_path: str) -> Optional[str]:
        """Video stabilizasyonu"""
        return self._apply_stabilization(video_path)
    
    def _sharpen_video(self, video_path: str) -> Optional[str]:
        """Video keskinlik artırma"""
        return self._apply_sharpening(video_path)
    
    def _add_grain(self, video_path: str) -> Optional[str]:
        """Video'ya grain ekle"""
        import subprocess
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"storage/outputs/advanced_ai/grain_{timestamp}.mp4"
        
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vf', 'noise=alls=20:allf=t',
            '-c:a', 'copy',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError:
            return None
