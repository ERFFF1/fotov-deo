#!/usr/bin/env python3
"""
İş Kuyruğu Görevleri
RQ tabanlı arka plan işlemleri
"""

import os
import sys
import json
import yaml
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Proje kök dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.database import get_db_manager
from core.config import get_config_manager, JobConfig
from pipelines.face import FacePipeline
from pipelines.body import BodyPipeline
from pipelines.environment import EnvironmentPipeline
from pipelines.color_audio import ColorAudioPipeline
from pipelines.talking import TalkingPipeline

class TaskManager:
    """Görev yöneticisi"""
    
    def __init__(self):
        self.db = get_db_manager()
        self.config = get_config_manager()
        
        # Pipeline'ları başlat
        self.face_pipeline = FacePipeline()
        self.body_pipeline = BodyPipeline()
        self.environment_pipeline = EnvironmentPipeline()
        self.color_audio_pipeline = ColorAudioPipeline()
        self.talking_pipeline = TalkingPipeline()
    
    def process_job(self, job_id: int) -> Dict[str, Any]:
        """Ana iş işleme fonksiyonu"""
        try:
            # İş bilgisini al
            job = self.db.get_job(job_id)
            if not job:
                return {'success': False, 'error': 'İş bulunamadı'}
            
            # İş durumunu güncelle
            self.db.update_job_status(job_id, 'running', 0, 'Başlatılıyor')
            
            # YAML konfigürasyonunu parse et
            config_data = yaml.safe_load(job.yaml_config)
            job_config = JobConfig(**config_data)
            
            # Konfigürasyonu doğrula
            errors = self.config.validate_config(job_config)
            if errors:
                error_msg = '; '.join(errors)
                self.db.update_job_status(job_id, 'failed', error_message=error_msg)
                return {'success': False, 'error': error_msg}
            
            # İş türüne göre pipeline'ı çalıştır
            result = self._execute_pipeline(job_id, job_config)
            
            if result['success']:
                # Başarılı sonuç
                self.db.set_job_output(
                    job_id, 
                    result['output_path'], 
                    result.get('metrics', {})
                )
                return result
            else:
                # Hata
                self.db.update_job_status(job_id, 'failed', error_message=result['error'])
                return result
                
        except Exception as e:
            error_msg = f'İş işleme hatası: {str(e)}'
            self.db.update_job_status(job_id, 'failed', error_message=error_msg)
            return {'success': False, 'error': error_msg}
    
    def _execute_pipeline(self, job_id: int, config: JobConfig) -> Dict[str, Any]:
        """Pipeline'ı çalıştır"""
        job_type = config.job_type
        
        # İlerleme güncelleme callback'i
        def update_progress(step: str, progress: float, completed_steps: int = None, total_steps: int = None):
            self.db.update_job_status(job_id, 'running', progress, step)
            if completed_steps is not None and total_steps is not None:
                self.db.update_job_progress(job_id, completed_steps, total_steps)
        
        try:
            if job_type in ['face_detect', 'face_recognize', 'face_enhance', 'face_swap']:
                return self.face_pipeline.execute(config, update_progress)
            
            elif job_type in ['body_segment', 'outfit_replace']:
                return self.body_pipeline.execute(config, update_progress)
            
            elif job_type in ['bg_replace', 'video_swap']:
                return self.environment_pipeline.execute(config, update_progress)
            
            elif job_type in ['color_grade', 'audio_enhance']:
                return self.color_audio_pipeline.execute(config, update_progress)
            
            elif job_type == 'talking_head':
                return self.talking_pipeline.execute(config, update_progress)
            
            else:
                return {'success': False, 'error': f'Desteklenmeyen iş türü: {job_type}'}
                
        except Exception as e:
            return {'success': False, 'error': f'Pipeline hatası: {str(e)}'}

# Global görev yöneticisi
task_manager = TaskManager()

# RQ görev fonksiyonları
def process_job_task(job_id: int) -> Dict[str, Any]:
    """RQ görev fonksiyonu - iş işleme"""
    return task_manager.process_job(job_id)

def cleanup_old_jobs(days: int = 7) -> Dict[str, Any]:
    """Eski işleri temizle"""
    try:
        # Bu fonksiyon daha sonra implement edilecek
        return {'success': True, 'cleaned_jobs': 0}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def validate_models() -> Dict[str, Any]:
    """Modelleri doğrula"""
    try:
        # Model doğrulama işlemi
        return {'success': True, 'valid_models': []}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Test fonksiyonu
def test_task() -> Dict[str, Any]:
    """Test görevi"""
    return {
        'success': True,
        'message': 'Test görevi başarılı',
        'timestamp': datetime.now().isoformat()
    }
