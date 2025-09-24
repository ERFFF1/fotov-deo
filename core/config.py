#!/usr/bin/env python3
"""
Konfigürasyon Yönetimi
YAML tabanlı konfigürasyon sistemi ve preset yönetimi
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class JobConfig(BaseModel):
    """İş konfigürasyonu modeli"""
    job_type: str = Field(..., description="İş türü")
    inputs: Dict[str, str] = Field(default_factory=dict, description="Giriş dosyaları")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parametreler")
    preset: Optional[str] = Field(None, description="Preset adı")
    consent_tag: str = Field("unknown", description="Rıza etiketi")
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ModelInfo(BaseModel):
    """Model bilgisi"""
    name: str
    version: str
    sha256: str
    path: str
    source_url: Optional[str] = None
    size_mb: Optional[float] = None
    last_updated: datetime = Field(default_factory=datetime.now)

class ConfigManager:
    """Konfigürasyon yöneticisi"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Preset dizini
        self.presets_dir = self.config_dir / "presets"
        self.presets_dir.mkdir(exist_ok=True)
        
        # Varsayılan konfigürasyon
        self.default_config = self._load_default_config()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Varsayılan konfigürasyonu yükle"""
        default_path = self.config_dir / "default.yaml"
        
        if default_path.exists():
            with open(default_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # Varsayılan konfigürasyonu oluştur
            default_config = {
                'system': {
                    'max_file_size_mb': 100,
                    'allowed_formats': {
                        'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
                        'video': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
                        'audio': ['.wav', '.mp3', '.aac', '.flac']
                    },
                    'storage': {
                        'uploads': 'storage/uploads',
                        'outputs': 'storage/outputs',
                        'artifacts': 'storage/artifacts',
                        'cache': 'storage/cache'
                    }
                },
                'models': {
                    'face_detection': {
                        'default': 'insightface',
                        'available': ['opencv', 'mediapipe', 'insightface']
                    },
                    'face_enhancement': {
                        'default': 'gfpgan',
                        'available': ['opencv', 'gfpgan', 'real_esrgan']
                    },
                    'face_swapping': {
                        'default': 'insightface',
                        'available': ['opencv', 'insightface']
                    },
                    'talking_head': {
                        'default': 'wav2lip',
                        'available': ['wav2lip', 'sadtalker', 'opencv']
                    }
                },
                'processing': {
                    'default_fps': 25,
                    'default_quality': 'high',
                    'gpu_enabled': True,
                    'max_workers': 4
                }
            }
            
            # Varsayılan konfigürasyonu kaydet
            with open(default_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
            
            return default_config
    
    def get_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Preset konfigürasyonunu al"""
        preset_path = self.presets_dir / f"{preset_name}.yaml"
        
        if not preset_path.exists():
            return None
        
        with open(preset_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def save_preset(self, preset_name: str, config: Dict[str, Any]) -> bool:
        """Preset konfigürasyonunu kaydet"""
        try:
            preset_path = self.presets_dir / f"{preset_name}.yaml"
            
            with open(preset_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            return True
        except Exception as e:
            print(f"Preset kaydetme hatası: {e}")
            return False
    
    def list_presets(self) -> List[str]:
        """Mevcut preset'leri listele"""
        presets = []
        for preset_file in self.presets_dir.glob("*.yaml"):
            presets.append(preset_file.stem)
        return sorted(presets)
    
    def create_job_config(self, job_type: str, inputs: Dict[str, str], 
                         params: Dict[str, Any] = None, preset: str = None) -> JobConfig:
        """İş konfigürasyonu oluştur"""
        if params is None:
            params = {}
        
        # Preset varsa, parametreleri birleştir
        if preset:
            preset_config = self.get_preset(preset)
            if preset_config:
                # Preset parametrelerini varsayılan olarak kullan
                if 'params' in preset_config:
                    preset_params = preset_config['params'].copy()
                    preset_params.update(params)  # Kullanıcı parametreleri öncelikli
                    params = preset_params
        
        return JobConfig(
            job_type=job_type,
            inputs=inputs,
            params=params,
            preset=preset
        )
    
    def validate_config(self, config: JobConfig) -> List[str]:
        """Konfigürasyonu doğrula"""
        errors = []
        
        # İş türü kontrolü
        valid_job_types = [
            'face_detect', 'face_recognize', 'face_enhance', 'face_swap',
            'video_swap', 'talking_head', 'color_grade', 'bg_replace',
            'body_segment', 'outfit_replace', 'audio_enhance'
        ]
        
        if config.job_type not in valid_job_types:
            errors.append(f"Geçersiz iş türü: {config.job_type}")
        
        # Giriş dosyaları kontrolü
        for input_type, file_path in config.inputs.items():
            if not os.path.exists(file_path):
                errors.append(f"Giriş dosyası bulunamadı: {input_type} -> {file_path}")
        
        # Rıza etiketi kontrolü
        valid_consent_tags = ['consented', 'demo', 'unknown']
        if config.consent_tag not in valid_consent_tags:
            errors.append(f"Geçersiz rıza etiketi: {config.consent_tag}")
        
        return errors
    
    def get_system_config(self) -> Dict[str, Any]:
        """Sistem konfigürasyonunu al"""
        return self.default_config.get('system', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Model konfigürasyonunu al"""
        return self.default_config.get('models', {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """İşleme konfigürasyonunu al"""
        return self.default_config.get('processing', {})

# Global konfigürasyon yöneticisi
config_manager = ConfigManager()

def get_config_manager() -> ConfigManager:
    """Konfigürasyon yöneticisini al"""
    return config_manager
