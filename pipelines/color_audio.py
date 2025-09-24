#!/usr/bin/env python3
"""
Color & Audio Pipeline
Renk düzeltme ve ses işleme pipeline'ı
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

class ColorAudioPipeline:
    """Renk ve ses işleme pipeline'ı"""
    
    def __init__(self):
        self.setup_directories()
    
    def setup_directories(self):
        """Gerekli klasörleri oluştur"""
        directories = [
            "storage/outputs/color_audio",
            "storage/artifacts/color_audio",
            "storage/cache/color_audio"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def execute(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Pipeline'ı çalıştır"""
        job_type = config.job_type
        
        if job_type == 'color_grade':
            return self._color_grade(config, progress_callback)
        elif job_type == 'audio_enhance':
            return self._enhance_audio(config, progress_callback)
        else:
            return {'success': False, 'error': f'Desteklenmeyen renk/ses işlemi: {job_type}'}
    
    def _color_grade(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Renk düzeltme"""
        try:
            progress_callback('Renk düzeltme başlatılıyor', 10)
            
            # Parametreleri al
            input_path = config.inputs.get('input')
            
            if not input_path or not os.path.exists(input_path):
                return {'success': False, 'error': 'Giriş dosyası bulunamadı'}
            
            progress_callback('Renk düzeltme yapılıyor', 50)
            
            # Renk düzeltme parametreleri
            saturation = config.params.get('saturation', 1.0)
            contrast = config.params.get('contrast', 1.0)
            brightness = config.params.get('brightness', 0.0)
            gamma = config.params.get('gamma', 1.0)
            
            # Görüntüyü yükle
            image = cv2.imread(input_path)
            
            # Renk düzeltme uygula
            # Saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
            # Contrast ve Brightness
            image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
            
            # Gamma correction
            if gamma != 1.0:
                lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
                image = cv2.LUT(image, lookup_table)
            
            # Sonuç görüntüsünü kaydet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"storage/outputs/color_audio/color_grade_{timestamp}.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, image)
            
            progress_callback('Renk düzeltme tamamlandı', 100)
            
            return {
                'success': True,
                'output_path': output_path,
                'metrics': {
                    'processing_time': datetime.now().isoformat(),
                    'saturation': saturation,
                    'contrast': contrast,
                    'brightness': brightness,
                    'gamma': gamma
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Renk düzeltme hatası: {str(e)}'}
    
    def _enhance_audio(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Ses iyileştirme"""
        try:
            progress_callback('Ses iyileştirme başlatılıyor', 10)
            
            # Parametreleri al
            audio_path = config.inputs.get('audio')
            
            if not audio_path or not os.path.exists(audio_path):
                return {'success': False, 'error': 'Ses dosyası bulunamadı'}
            
            progress_callback('Ses iyileştirme yapılıyor', 50)
            
            # Ses iyileştirme parametreleri
            normalize = config.params.get('normalize', True)
            remove_noise = config.params.get('remove_noise', False)
            loudness_target = config.params.get('loudness_target', -16)  # LUFS
            
            # FFmpeg ile ses iyileştirme
            import subprocess
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"storage/outputs/color_audio/audio_enhanced_{timestamp}.wav"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # FFmpeg komutu oluştur
            cmd = ['ffmpeg', '-y', '-i', audio_path]
            
            # Ses filtreleri
            filters = []
            
            if remove_noise:
                filters.append('afftdn=nf=-25')
            
            if normalize:
                filters.append(f'loudnorm=I={loudness_target}:TP=-1.5:LRA=11')
            
            if filters:
                cmd.extend(['-af', ','.join(filters)])
            
            cmd.append(output_path)
            
            # FFmpeg çalıştır
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {'success': False, 'error': f'FFmpeg hatası: {result.stderr}'}
            
            progress_callback('Ses iyileştirme tamamlandı', 100)
            
            return {
                'success': True,
                'output_path': output_path,
                'metrics': {
                    'processing_time': datetime.now().isoformat(),
                    'normalize': normalize,
                    'remove_noise': remove_noise,
                    'loudness_target': loudness_target
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Ses iyileştirme hatası: {str(e)}'}
