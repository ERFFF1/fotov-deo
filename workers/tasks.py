#!/usr/bin/env python3
"""
Face/Video AI Studio - Worker Tasks
Ağır ML işlemlerini yapan task'lar
"""

import os
import sys
import time
import traceback
from typing import Dict, Any

# Proje kök dizinini Python path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def process_job_task(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ana işlem task'ı - Worker'da çalışır
    """
    try:
        print(f"🎯 İş başlatılıyor: {job_data}")
        
        job_type = job_data.get("job_type", "unknown")
        
        # İş türüne göre pipeline'ı seç
        if job_type == "face_swap":
            result = process_face_swap(job_data)
        elif job_type == "talking_head":
            result = process_talking_head(job_data)
        elif job_type == "color_grade":
            result = process_color_grade(job_data)
        else:
            result = {"error": f"Bilinmeyen iş türü: {job_type}"}
        
        print(f"✅ İş tamamlandı: {result}")
        return result
        
    except Exception as e:
        error_msg = f"❌ İş hatası: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return {"error": error_msg}

def process_face_swap(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Face swap işlemi"""
    try:
        # Ağır import'lar burada yapılır
        from pipelines.face import FacePipeline
        from core.config import JobConfig
        
        print("🔄 Face swap pipeline başlatılıyor...")
        
        # Pipeline'ı çalıştır
        pipeline = FacePipeline()
        config = JobConfig(
            job_type="face_swap",
            inputs=job_data.get("inputs", {}),
            params=job_data.get("params", {})
        )
        
        result = pipeline.execute(config, progress_callback=None)
        
        return {
            "success": True,
            "job_type": "face_swap",
            "result": result,
            "processing_time": time.time()
        }
        
    except Exception as e:
        return {"error": f"Face swap hatası: {str(e)}"}

def process_talking_head(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Talking head işlemi"""
    try:
        from pipelines.talking import TalkingPipeline
        from core.config import JobConfig
        
        print("🗣️ Talking head pipeline başlatılıyor...")
        
        pipeline = TalkingPipeline()
        config = JobConfig(
            job_type="talking_head",
            inputs=job_data.get("inputs", {}),
            params=job_data.get("params", {})
        )
        
        result = pipeline.execute(config, progress_callback=None)
        
        return {
            "success": True,
            "job_type": "talking_head",
            "result": result,
            "processing_time": time.time()
        }
        
    except Exception as e:
        return {"error": f"Talking head hatası: {str(e)}"}

def process_color_grade(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Color grade işlemi"""
    try:
        from pipelines.color_audio import ColorAudioPipeline
        from core.config import JobConfig
        
        print("🎨 Color grade pipeline başlatılıyor...")
        
        pipeline = ColorAudioPipeline()
        config = JobConfig(
            job_type="color_grade",
            inputs=job_data.get("inputs", {}),
            params=job_data.get("params", {})
        )
        
        result = pipeline.execute(config, progress_callback=None)
        
        return {
            "success": True,
            "job_type": "color_grade",
            "result": result,
            "processing_time": time.time()
        }
        
    except Exception as e:
        return {"error": f"Color grade hatası: {str(e)}"}

def test_worker():
    """Worker test fonksiyonu"""
    test_job = {
        "job_type": "face_swap",
        "inputs": {
            "source_image": "test.jpg",
            "target_video": "test.mp4"
        },
        "params": {
            "quality": "high"
        }
    }
    
    result = process_job_task(test_job)
    print(f"Test sonucu: {result}")
    return result

if __name__ == "__main__":
    # Test için
    test_worker()