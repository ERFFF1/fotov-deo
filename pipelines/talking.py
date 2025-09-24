#!/usr/bin/env python3
"""
Talking Pipeline
Konuşan kafa işleme pipeline'ı
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from datetime import datetime

# Proje kök dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import JobConfig

class TalkingPipeline:
    """Konuşan kafa işleme pipeline'ı"""
    
    def __init__(self):
        self.setup_directories()
    
    def setup_directories(self):
        """Gerekli klasörleri oluştur"""
        directories = [
            "storage/outputs/talking",
            "storage/artifacts/talking",
            "storage/cache/talking"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def execute(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Pipeline'ı çalıştır"""
        job_type = config.job_type
        
        if job_type == 'talking_head':
            return self._generate_talking_head(config, progress_callback)
        else:
            return {'success': False, 'error': f'Desteklenmeyen konuşan kafa işlemi: {job_type}'}
    
    def _generate_talking_head(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Konuşan kafa üretimi"""
        try:
            progress_callback('Konuşan kafa üretimi başlatılıyor', 10)
            
            # Parametreleri al
            image_path = config.inputs.get('image')
            audio_path = config.inputs.get('audio')
            method = config.params.get('method', 'wav2lip')
            
            if not image_path or not os.path.exists(image_path):
                return {'success': False, 'error': 'Görüntü dosyası bulunamadı'}
            
            if not audio_path or not os.path.exists(audio_path):
                return {'success': False, 'error': 'Ses dosyası bulunamadı'}
            
            progress_callback('Konuşan kafa sistemi başlatılıyor', 30)
            
            # Konuşan kafa üretici
            from talking_head import TalkingHeadGenerator
            
            generator = TalkingHeadGenerator(method=method)
            
            # Çıktı dosya yolu
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"storage/outputs/talking/talking_head_{timestamp}.mp4"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            progress_callback('Konuşan kafa üretiliyor', 60)
            
            # Konuşan kafa üret
            result = generator.generate_talking_head(image_path, audio_path, output_path)
            
            if not result['success']:
                return result
            
            progress_callback('Konuşan kafa üretimi tamamlandı', 100)
            
            return {
                'success': True,
                'output_path': result['output_path'],
                'metrics': {
                    'processing_time': datetime.now().isoformat(),
                    'talking_method': method,
                    'generation_info': result
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Konuşan kafa üretimi hatası: {str(e)}'}
