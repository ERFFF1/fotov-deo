#!/usr/bin/env python3
"""
Temporal Consistency Pipeline
Optical-flow tabanlı zamansal tutarlılık ve renk uyumu
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json

class TemporalConsistencyPipeline:
    """Zamansal tutarlılık pipeline'ı"""
    
    def __init__(self):
        self.setup_directories()
        self.prev_frame = None
        self.face_tracker = None
        self.optical_flow = None
        
    def setup_directories(self):
        """Gerekli klasörleri oluştur"""
        directories = [
            "storage/outputs/temporal",
            "storage/artifacts/temporal",
            "storage/cache/temporal"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def detect_and_track_faces(self, frame: np.ndarray, frame_id: int) -> Dict[str, Any]:
        """Yüz tespiti ve takibi"""
        # Yüz tespiti
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_data = {
            'frame_id': frame_id,
            'faces': [],
            'tracking_data': {}
        }
        
        for i, (x, y, w, h) in enumerate(faces):
            face_info = {
                'id': i,
                'bbox': (x, y, w, h),
                'center': (x + w//2, y + h//2),
                'confidence': 1.0
            }
            face_data['faces'].append(face_info)
        
        return face_data
    
    def match_color_reinhard(self, src_bgr: np.ndarray, ref_bgr: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """Reinhard renk eşleme"""
        src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype("float32")
        ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype("float32")
        
        for c in range(3):
            s_mean, s_std = src_lab[..., c].mean(), src_lab[..., c].std() + 1e-6
            r_mean, r_std = ref_lab[..., c].mean(), ref_lab[..., c].std() + 1e-6
            src_lab[..., c] = (src_lab[..., c] - s_mean) * (r_std / s_std) + r_mean
        
        out = cv2.cvtColor(np.clip(src_lab, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)
        return out
    
    def temporal_smooth_ema(self, curr: np.ndarray, beta: float = 0.7) -> np.ndarray:
        """Temporal smoothing with EMA"""
        if self.prev_frame is None:
            self.prev_frame = curr.copy()
            return curr
        
        blended = cv2.addWeighted(curr, beta, self.prev_frame, 1 - beta, 0)
        self.prev_frame = blended
        return blended
    
    def calculate_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        """Optical flow hesaplama"""
        flow = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray,
            None, None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        return flow
    
    def apply_temporal_consistency(self, frame: np.ndarray, face_data: Dict[str, Any], 
                                 reference_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Zamansal tutarlılık uygula"""
        result = frame.copy()
        
        # Renk eşleme (referans frame varsa)
        if reference_frame is not None:
            result = self.match_color_reinhard(result, reference_frame)
        
        # Temporal smoothing
        result = self.temporal_smooth_ema(result)
        
        return result
    
    def process_video_temporal(self, input_path: str, output_path: str, 
                             reference_face_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Video temporal consistency işleme"""
        start_time = datetime.now()
        
        # Video aç
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Çıktı video yazıcı
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Referans yüz
        reference_face = cv2.imread(reference_face_path)
        
        # Metrikler
        metrics = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'face_detection_rate': 0.0,
            'temporal_consistency_score': 0.0,
            'color_matching_score': 0.0
        }
        
        frame_id = 0
        prev_gray = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Yüz tespiti ve takibi
            face_data = self.detect_and_track_faces(frame, frame_id)
            
            # Optical flow hesaplama
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = self.calculate_optical_flow(prev_gray, curr_gray)
            
            # Temporal consistency uygula
            processed_frame = self.apply_temporal_consistency(
                frame, face_data, reference_face
            )
            
            # Çıktıya yaz
            out.write(processed_frame)
            
            # Metrikleri güncelle
            metrics['processed_frames'] += 1
            if face_data['faces']:
                metrics['face_detection_rate'] += 1
            
            frame_id += 1
            prev_gray = curr_gray
            
            # İlerleme göster
            if frame_id % 30 == 0:
                progress = (frame_id / total_frames) * 100
                print(f"Temporal processing: {progress:.1f}% - Frame {frame_id}/{total_frames}")
        
        # Temizlik
        cap.release()
        out.release()
        
        # Metrikleri hesapla
        metrics['face_detection_rate'] = metrics['face_detection_rate'] / metrics['processed_frames']
        metrics['processing_time'] = (datetime.now() - start_time).total_seconds()
        metrics['fps_processed'] = metrics['processed_frames'] / metrics['processing_time']
        
        return {
            'success': True,
            'output_path': output_path,
            'metrics': metrics,
            'processing_time': metrics['processing_time']
        }
    
    def calculate_temporal_jitter(self, flow_data: list) -> float:
        """Temporal jitter hesaplama (optical flow std)"""
        if not flow_data:
            return 0.0
        
        # Flow vektörlerinin standart sapması
        flow_vectors = np.array(flow_data)
        jitter = np.std(flow_vectors)
        return float(jitter)
    
    def calculate_color_delta_e(self, before: np.ndarray, after: np.ndarray) -> float:
        """Color ΔE (CIELAB) hesaplama"""
        # BGR'den LAB'ye dönüştür
        before_lab = cv2.cvtColor(before, cv2.COLOR_BGR2LAB)
        after_lab = cv2.cvtColor(after, cv2.COLOR_BGR2LAB)
        
        # ΔE hesapla
        delta_e = np.sqrt(np.sum((before_lab - after_lab) ** 2, axis=2))
        return float(np.mean(delta_e))
    
    def calculate_face_alignment_score(self, landmarks: np.ndarray, 
                                     reference_landmarks: np.ndarray) -> float:
        """Face alignment score (landmark reprojection error)"""
        if landmarks is None or reference_landmarks is None:
            return 0.0
        
        # Reprojection error
        error = np.mean(np.sqrt(np.sum((landmarks - reference_landmarks) ** 2, axis=1)))
        return float(error)

# Global instance
_temporal_pipeline = None

def get_temporal_pipeline() -> TemporalConsistencyPipeline:
    """Global temporal pipeline instance"""
    global _temporal_pipeline
    if _temporal_pipeline is None:
        _temporal_pipeline = TemporalConsistencyPipeline()
    return _temporal_pipeline
