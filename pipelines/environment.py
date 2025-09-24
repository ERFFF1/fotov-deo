#!/usr/bin/env python3
"""
Environment Pipeline
Ortam ve arka plan işleme pipeline'ı
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from datetime import datetime

# Proje kök dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import JobConfig

class EnvironmentPipeline:
    """Ortam işleme pipeline'ı"""
    
    def __init__(self):
        self.setup_directories()
    
    def setup_directories(self):
        """Gerekli klasörleri oluştur"""
        directories = [
            "storage/outputs/environment",
            "storage/artifacts/environment",
            "storage/cache/environment"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def execute(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Pipeline'ı çalıştır"""
        job_type = config.job_type
        
        if job_type == 'bg_replace':
            return self._replace_background(config, progress_callback)
        elif job_type == 'video_swap':
            return self._swap_video_faces(config, progress_callback)
        else:
            return {'success': False, 'error': f'Desteklenmeyen ortam işlemi: {job_type}'}
    
    def _replace_background(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Arka plan değiştirme"""
        try:
            progress_callback('Arka plan değiştirme başlatılıyor', 10)
            
            # Parametreleri al
            image_path = config.inputs.get('image')
            background_path = config.inputs.get('background')
            
            if not image_path or not os.path.exists(image_path):
                return {'success': False, 'error': 'Görüntü dosyası bulunamadı'}
            
            if not background_path or not os.path.exists(background_path):
                return {'success': False, 'error': 'Arka plan dosyası bulunamadı'}
            
            progress_callback('Arka plan değiştirme yapılıyor', 50)
            
            # Basit arka plan değiştirme (demo)
            image = cv2.imread(image_path)
            background = cv2.imread(background_path)
            
            # Basit chroma key benzeri işlem
            h, w = image.shape[:2]
            background_resized = cv2.resize(background, (w, h))
            
            # Basit maskeleme (yeşil tonları için)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Maske ile birleştirme
            result = image.copy()
            result[mask > 0] = background_resized[mask > 0]
            
            # Sonuç görüntüsünü kaydet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"storage/outputs/environment/bg_replace_{timestamp}.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, result)
            
            progress_callback('Arka plan değiştirme tamamlandı', 100)
            
            return {
                'success': True,
                'output_path': output_path,
                'metrics': {
                    'processing_time': datetime.now().isoformat(),
                    'replacement_method': 'chroma_key'
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Arka plan değiştirme hatası: {str(e)}'}
    
    def _swap_video_faces(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Video yüz değiştirme"""
        try:
            progress_callback('Video yüz değiştirme başlatılıyor', 10)
            
            # Parametreleri al
            video_path = config.inputs.get('video')
            face_path = config.inputs.get('source_face')
            
            if not video_path or not os.path.exists(video_path):
                return {'success': False, 'error': 'Video dosyası bulunamadı'}
            
            if not face_path or not os.path.exists(face_path):
                return {'success': False, 'error': 'Yüz dosyası bulunamadı'}
            
            progress_callback('Video işleme başlatılıyor', 30)
            
            # Video işleme pipeline'ını kullan
            from video_processor import VideoProcessor
            
            method = config.params.get('method', 'insightface')
            processor = VideoProcessor(method=method)
            
            # Video yüz değiştirme yap
            result = processor.swap_face_in_video(video_path, face_path)
            
            if not result['success']:
                return result
            
            progress_callback('Video yüz değiştirme tamamlandı', 100)
            
            return {
                'success': True,
                'output_path': result['output_path'],
                'metrics': {
                    'processing_time': datetime.now().isoformat(),
                    'video_method': method,
                    'frames_processed': result['processed_frames']
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Video yüz değiştirme hatası: {str(e)}'}
