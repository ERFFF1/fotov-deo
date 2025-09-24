#!/usr/bin/env python3
"""
Body Pipeline
Beden ve kıyafet işleme pipeline'ı
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

class BodyPipeline:
    """Beden işleme pipeline'ı"""
    
    def __init__(self):
        self.setup_directories()
    
    def setup_directories(self):
        """Gerekli klasörleri oluştur"""
        directories = [
            "storage/outputs/body",
            "storage/artifacts/body",
            "storage/cache/body"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def execute(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Pipeline'ı çalıştır"""
        job_type = config.job_type
        
        if job_type == 'body_segment':
            return self._segment_body(config, progress_callback)
        elif job_type == 'outfit_replace':
            return self._replace_outfit(config, progress_callback)
        else:
            return {'success': False, 'error': f'Desteklenmeyen beden işlemi: {job_type}'}
    
    def _segment_body(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Beden segmentasyonu"""
        try:
            progress_callback('Beden segmentasyonu başlatılıyor', 10)
            
            # Parametreleri al
            image_path = config.inputs.get('image')
            
            if not image_path or not os.path.exists(image_path):
                return {'success': False, 'error': 'Görüntü dosyası bulunamadı'}
            
            progress_callback('Beden segmentasyonu yapılıyor', 50)
            
            # Basit beden segmentasyonu (OpenCV ile)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Basit threshold ile segmentasyon
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Sonuç görüntüsünü kaydet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"storage/outputs/body/segmentation_{timestamp}.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, mask)
            
            progress_callback('Beden segmentasyonu tamamlandı', 100)
            
            return {
                'success': True,
                'output_path': output_path,
                'metrics': {
                    'processing_time': datetime.now().isoformat(),
                    'segmentation_method': 'opencv_threshold'
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Beden segmentasyonu hatası: {str(e)}'}
    
    def _replace_outfit(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Kıyafet değiştirme"""
        try:
            progress_callback('Kıyafet değiştirme başlatılıyor', 10)
            
            # Parametreleri al
            image_path = config.inputs.get('image')
            outfit_path = config.inputs.get('outfit')
            
            if not image_path or not os.path.exists(image_path):
                return {'success': False, 'error': 'Görüntü dosyası bulunamadı'}
            
            if not outfit_path or not os.path.exists(outfit_path):
                return {'success': False, 'error': 'Kıyafet dosyası bulunamadı'}
            
            progress_callback('Kıyafet değiştirme yapılıyor', 50)
            
            # Basit kıyafet değiştirme (demo)
            image = cv2.imread(image_path)
            outfit = cv2.imread(outfit_path)
            
            # Basit overlay işlemi
            h, w = image.shape[:2]
            outfit_resized = cv2.resize(outfit, (w, h))
            
            # Basit karıştırma
            result = cv2.addWeighted(image, 0.7, outfit_resized, 0.3, 0)
            
            # Sonuç görüntüsünü kaydet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"storage/outputs/body/outfit_replace_{timestamp}.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, result)
            
            progress_callback('Kıyafet değiştirme tamamlandı', 100)
            
            return {
                'success': True,
                'output_path': output_path,
                'metrics': {
                    'processing_time': datetime.now().isoformat(),
                    'replacement_method': 'opencv_blend'
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Kıyafet değiştirme hatası: {str(e)}'}
