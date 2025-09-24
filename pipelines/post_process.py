#!/usr/bin/env python3
"""
Post-Process Pipeline
Konuşan kafa ve video post-prodüksiyon preset'leri
"""

import cv2
import numpy as np
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

class PostProcessPipeline:
    """Post-prodüksiyon pipeline'ı"""
    
    def __init__(self):
        self.setup_directories()
        self.presets = self._load_presets()
        
    def setup_directories(self):
        """Gerekli klasörleri oluştur"""
        directories = [
            "storage/outputs/post_process",
            "storage/artifacts/post_process",
            "storage/cache/post_process"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _load_presets(self) -> Dict[str, Dict[str, Any]]:
        """Post-process preset'lerini yükle"""
        return {
            "talking_head_pro": {
                "name": "Talking Head Pro",
                "description": "Stabilize + Sharpen + Grain + Audio Normalize",
                "video": {
                    "stabilize": True,
                    "sharpen": True,
                    "grain": True,
                    "grain_strength": 0.1,
                    "sharpen_strength": 0.3
                },
                "audio": {
                    "loudness_normalize": True,
                    "target_lufs": -16,
                    "highpass": 80,
                    "noise_gate": True,
                    "limiter": -1
                }
            },
            "cinematic_grade": {
                "name": "Cinematic Grade",
                "description": "Film look + Color grading + Audio enhancement",
                "video": {
                    "lut": "rec709.cube",
                    "contrast": 1.1,
                    "saturation": 1.05,
                    "gamma": 0.9,
                    "vignette": True
                },
                "audio": {
                    "loudness_normalize": True,
                    "target_lufs": -23,
                    "highpass": 60,
                    "lowpass": 18000
                }
            },
            "social_media": {
                "name": "Social Media",
                "description": "Optimized for social platforms",
                "video": {
                    "stabilize": True,
                    "sharpen": True,
                    "contrast": 1.15,
                    "saturation": 1.1
                },
                "audio": {
                    "loudness_normalize": True,
                    "target_lufs": -14,
                    "compression": True
                }
            }
        }
    
    def apply_video_stabilization(self, input_path: str, output_path: str, 
                                strength: float = 0.5) -> Dict[str, Any]:
        """Video stabilizasyonu"""
        try:
            # FFmpeg ile stabilizasyon
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vf', f'vidstabdetect=stepsize=6:shakiness=8:accuracy=9:result=transforms.trf',
                '-f', 'null', '-'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Stabilizasyon uygula
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vf', f'vidstabtransform=input=transforms.trf:smoothing=30:optalgo=gauss:maxshift=-1:maxangle=0.1:crop=black',
                '-c:a', 'copy',
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            return {'success': True, 'output_path': output_path}
            
        except subprocess.CalledProcessError as e:
            return {'success': False, 'error': str(e)}
    
    def apply_sharpening(self, frame: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """Görüntü keskinleştirme"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * strength
        
        sharpened = cv2.filter2D(frame, -1, kernel)
        return cv2.addWeighted(frame, 1 - strength, sharpened, strength, 0)
    
    def apply_film_grain(self, frame: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Film grain efekti"""
        noise = np.random.normal(0, strength * 25, frame.shape).astype(np.uint8)
        return cv2.add(frame, noise)
    
    def apply_color_grading(self, frame: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Renk düzeltme"""
        result = frame.copy().astype(np.float32)
        
        # Kontrast
        if 'contrast' in params:
            contrast = params['contrast']
            result = result * contrast
            result = np.clip(result, 0, 255)
        
        # Doygunluk
        if 'saturation' in params:
            saturation = params['saturation']
            hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Gamma
        if 'gamma' in params:
            gamma = params['gamma']
            result = np.power(result / 255.0, gamma) * 255.0
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_audio_processing(self, input_path: str, output_path: str, 
                             audio_params: Dict[str, Any]) -> Dict[str, Any]:
        """Ses işleme"""
        try:
            cmd = ['ffmpeg', '-i', input_path]
            
            # Audio filter chain
            audio_filters = []
            
            # Highpass filter
            if audio_params.get('highpass'):
                audio_filters.append(f"highpass=f={audio_params['highpass']}")
            
            # Noise gate
            if audio_params.get('noise_gate'):
                audio_filters.append("anlmdn=s=0.001:p=0.01")
            
            # Loudness normalization
            if audio_params.get('loudness_normalize'):
                target_lufs = audio_params.get('target_lufs', -16)
                audio_filters.append(f"loudnorm=I={target_lufs}:TP=-1:LRA=11")
            
            # Limiter
            if audio_params.get('limiter'):
                limiter_db = audio_params.get('limiter', -1)
                audio_filters.append(f"alimiter=level_in=1:level_out=1:limit={limiter_db}")
            
            # Filter chain'i uygula
            if audio_filters:
                cmd.extend(['-af', ':'.join(audio_filters)])
            
            cmd.extend(['-c:v', 'copy', output_path])
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            return {'success': True, 'output_path': output_path}
            
        except subprocess.CalledProcessError as e:
            return {'success': False, 'error': str(e)}
    
    def apply_preset(self, input_path: str, output_path: str, 
                    preset_name: str) -> Dict[str, Any]:
        """Preset uygula"""
        if preset_name not in self.presets:
            return {'success': False, 'error': f'Preset {preset_name} bulunamadı'}
        
        preset = self.presets[preset_name]
        start_time = datetime.now()
        
        try:
            # Video işleme
            if 'video' in preset:
                video_params = preset['video']
                
                # Stabilizasyon
                if video_params.get('stabilize'):
                    temp_path = str(Path(output_path).with_suffix('.temp.mp4'))
                    result = self.apply_video_stabilization(input_path, temp_path)
                    if not result['success']:
                        return result
                    input_path = temp_path
                
                # Frame-by-frame işleme
                cap = cv2.VideoCapture(input_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Keskinleştirme
                    if video_params.get('sharpen'):
                        frame = self.apply_sharpening(frame, video_params.get('sharpen_strength', 0.3))
                    
                    # Film grain
                    if video_params.get('grain'):
                        frame = self.apply_film_grain(frame, video_params.get('grain_strength', 0.1))
                    
                    # Renk düzeltme
                    frame = self.apply_color_grading(frame, video_params)
                    
                    out.write(frame)
                
                cap.release()
                out.release()
            
            # Ses işleme
            if 'audio' in preset:
                audio_result = self.apply_audio_processing(input_path, output_path, preset['audio'])
                if not audio_result['success']:
                    return audio_result
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'output_path': output_path,
                'preset_name': preset_name,
                'processing_time': processing_time,
                'applied_effects': list(preset.keys())
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def list_presets(self) -> List[Dict[str, Any]]:
        """Preset listesi"""
        return [
            {
                'name': name,
                'description': preset['description'],
                'effects': list(preset.keys())
            }
            for name, preset in self.presets.items()
        ]
    
    def create_custom_preset(self, name: str, description: str, 
                           video_params: Dict[str, Any], 
                           audio_params: Dict[str, Any]) -> Dict[str, Any]:
        """Özel preset oluştur"""
        self.presets[name] = {
            'name': name,
            'description': description,
            'video': video_params,
            'audio': audio_params
        }
        
        # Preset'i dosyaya kaydet
        preset_path = Path("configs/presets") / f"{name}.yaml"
        preset_path.parent.mkdir(parents=True, exist_ok=True)
        
        import yaml
        with open(preset_path, 'w') as f:
            yaml.dump(self.presets[name], f, default_flow_style=False)
        
        return {'success': True, 'preset_name': name}

# Global instance
_post_process_pipeline = None

def get_post_process_pipeline() -> PostProcessPipeline:
    """Global post-process pipeline instance"""
    global _post_process_pipeline
    if _post_process_pipeline is None:
        _post_process_pipeline = PostProcessPipeline()
    return _post_process_pipeline
