#!/usr/bin/env python3
"""
Face Pipeline
Yüz işleme pipeline'ı - tespit, tanıma, iyileştirme, değiştirme
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

# Mevcut modülleri import et
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from face_enhancer import FaceEnhancer
from face_swapper import FaceSwapper
from core.config import JobConfig

class FacePipeline:
    """Yüz işleme pipeline'ı"""
    
    def __init__(self):
        self.setup_directories()
    
    def setup_directories(self):
        """Gerekli klasörleri oluştur"""
        directories = [
            "storage/outputs/face",
            "storage/artifacts/face",
            "storage/cache/face"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def execute(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Pipeline'ı çalıştır"""
        job_type = config.job_type
        
        if job_type == 'face_detect':
            return self._detect_faces(config, progress_callback)
        elif job_type == 'face_recognize':
            return self._recognize_faces(config, progress_callback)
        elif job_type == 'face_enhance':
            return self._enhance_faces(config, progress_callback)
        elif job_type == 'face_swap':
            return self._swap_faces(config, progress_callback)
        else:
            return {'success': False, 'error': f'Desteklenmeyen yüz işlemi: {job_type}'}
    
    def _detect_faces(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Yüz tespiti"""
        try:
            progress_callback('Yüz tespiti başlatılıyor', 10)
            
            # Parametreleri al
            image_path = config.inputs.get('image')
            method = config.params.get('detector', 'opencv')
            save_faces = config.params.get('save_faces', False)
            
            if not image_path or not os.path.exists(image_path):
                return {'success': False, 'error': 'Görüntü dosyası bulunamadı'}
            
            progress_callback('Yüz tespit sistemi başlatılıyor', 20)
            
            # Yüz tespiti yap
            detector = FaceDetector(method=method)
            result = detector.detect_faces(image_path)
            
            if result is None:
                return {'success': False, 'error': 'Yüz tespiti başarısız'}
            
            faces_data, original_image = result
            
            progress_callback('Yüzler işleniyor', 60)
            
            # Sonuç görüntüsünü kaydet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"storage/outputs/face/detection_{timestamp}.jpg"
            result_image = detector.draw_detections(original_image, faces_data)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, result_image)
            
            # Yüzleri kaydet (isteğe bağlı)
            saved_faces = []
            if save_faces:
                saved_faces = detector.save_faces(faces_data, original_image, "storage/artifacts/face")
            
            progress_callback('Yüz tespiti tamamlandı', 100)
            
            return {
                'success': True,
                'output_path': output_path,
                'faces_count': faces_data['faces_count'],
                'saved_faces': saved_faces,
                'faces_data': faces_data,
                'metrics': {
                    'detection_method': method,
                    'processing_time': datetime.now().isoformat(),
                    'faces_detected': faces_data['faces_count']
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Yüz tespiti hatası: {str(e)}'}
    
    def _recognize_faces(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Yüz tanıma"""
        try:
            progress_callback('Yüz tanıma başlatılıyor', 10)
            
            # Parametreleri al
            image1_path = config.inputs.get('image1')
            image2_path = config.inputs.get('image2')
            method = config.params.get('method', 'face_recognition')
            
            if not image1_path or not image2_path:
                return {'success': False, 'error': 'İki görüntü dosyası gerekli'}
            
            if not os.path.exists(image1_path) or not os.path.exists(image2_path):
                return {'success': False, 'error': 'Görüntü dosyaları bulunamadı'}
            
            progress_callback('Yüz tanıma sistemi başlatılıyor', 30)
            
            # Yüz tanıma yap
            recognizer = FaceRecognizer(method=method)
            result = recognizer.compare_faces(image1_path, image2_path)
            
            if not result['success']:
                return result
            
            progress_callback('Yüz tanıma tamamlandı', 100)
            
            # Sonuç dosyasını kaydet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = f"storage/outputs/face/recognition_{timestamp}.json"
            Path(result_path).parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            return {
                'success': True,
                'output_path': result_path,
                'similarity': result['similarity'],
                'is_same_person': result['is_same_person'],
                'confidence': result.get('confidence', 'N/A'),
                'metrics': {
                    'recognition_method': method,
                    'processing_time': datetime.now().isoformat(),
                    'similarity_score': result['similarity']
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Yüz tanıma hatası: {str(e)}'}
    
    def _enhance_faces(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Yüz iyileştirme"""
        try:
            progress_callback('Yüz iyileştirme başlatılıyor', 10)
            
            # Parametreleri al
            image_path = config.inputs.get('image')
            method = config.params.get('method', 'gfpgan')
            
            if not image_path or not os.path.exists(image_path):
                return {'success': False, 'error': 'Görüntü dosyası bulunamadı'}
            
            progress_callback('Yüz iyileştirme sistemi başlatılıyor', 30)
            
            # Yüz iyileştirme yap
            enhancer = FaceEnhancer(method=method)
            result = enhancer.enhance_face(image_path)
            
            if not result['success']:
                return result
            
            progress_callback('Yüz iyileştirme tamamlandı', 100)
            
            return {
                'success': True,
                'output_path': result['output_path'],
                'quality_score': result['quality_score'],
                'metrics': {
                    'enhancement_method': method,
                    'processing_time': datetime.now().isoformat(),
                    'quality_improvement': result['quality_score']
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Yüz iyileştirme hatası: {str(e)}'}
    
    def _swap_faces(self, config: JobConfig, progress_callback: Callable) -> Dict[str, Any]:
        """Yüz değiştirme"""
        try:
            progress_callback('Yüz değiştirme başlatılıyor', 10)
            
            # Parametreleri al
            source_path = config.inputs.get('source_face')
            target_path = config.inputs.get('target_image')
            method = config.params.get('method', 'insightface')
            
            if not source_path or not target_path:
                return {'success': False, 'error': 'Kaynak ve hedef görüntü gerekli'}
            
            if not os.path.exists(source_path) or not os.path.exists(target_path):
                return {'success': False, 'error': 'Görüntü dosyaları bulunamadı'}
            
            progress_callback('Yüz değiştirme sistemi başlatılıyor', 30)
            
            # Yüz değiştirme yap
            swapper = FaceSwapper(method=method)
            result = swapper.swap_faces(source_path, target_path)
            
            if not result['success']:
                return result
            
            progress_callback('Yüz değiştirme tamamlandı', 100)
            
            return {
                'success': True,
                'output_path': result['output_path'],
                'swapped_faces': result['swapped_faces'],
                'metrics': {
                    'swapping_method': method,
                    'processing_time': datetime.now().isoformat(),
                    'faces_swapped': result['swapped_faces']
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Yüz değiştirme hatası: {str(e)}'}
