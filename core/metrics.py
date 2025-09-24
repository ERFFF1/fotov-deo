#!/usr/bin/env python3
"""
Objective Metrics Package
Kalite ölçümü ve metrik toplama sistemi
"""

import cv2
import numpy as np
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Dlib bulunamadı. Basit landmark tespiti kullanılacak.")

try:
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Scipy bulunamadı. Basit mesafe hesaplaması kullanılacak.")

class ObjectiveMetrics:
    """Objective metrik hesaplama sınıfı"""
    
    def __init__(self):
        self.setup_directories()
        self.face_predictor = None
        self.landmark_model = None
        
    def setup_directories(self):
        """Gerekli klasörleri oluştur"""
        directories = [
            "storage/artifacts/metrics",
            "storage/logs/metrics"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_face_landmark_model(self):
        """Yüz landmark modelini yükle"""
        if not DLIB_AVAILABLE:
            return False
            
        try:
            # Dlib 68-point landmark model
            model_path = "models/shape_predictor_68_face_landmarks.dat"
            if Path(model_path).exists():
                self.face_predictor = dlib.shape_predictor(model_path)
                return True
        except Exception as e:
            print(f"Landmark model yükleme hatası: {e}")
        return False
    
    def calculate_face_alignment_score(self, image: np.ndarray, 
                                     reference_landmarks: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Face alignment score hesaplama"""
        try:
            # Yüz tespiti
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {'alignment_score': 0.0, 'landmark_count': 0}
            
            # İlk yüzü al
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Landmark tespiti
            if self.face_predictor is None:
                self.load_face_landmark_model()
            
            if self.face_predictor is not None:
                # Dlib ile landmark tespiti
                rect = dlib.rectangle(x, y, x+w, y+h)
                landmarks = self.face_predictor(gray, rect)
                
                # Landmark koordinatlarını al
                landmark_points = np.array([[p.x, p.y] for p in landmarks.parts()])
                
                # Referans landmark'lar varsa karşılaştır
                if reference_landmarks is not None:
                    # Procrustes analizi ile alignment
                    aligned_landmarks = self._procrustes_analysis(landmark_points, reference_landmarks)
                    reprojection_error = np.mean(np.sqrt(np.sum((aligned_landmarks - reference_landmarks) ** 2, axis=1)))
                    alignment_score = max(0, 1 - (reprojection_error / 50))  # Normalize
                else:
                    # Standart landmark dağılımı kontrolü
                    alignment_score = self._calculate_landmark_quality(landmark_points)
                
                return {
                    'alignment_score': float(alignment_score),
                    'landmark_count': len(landmark_points),
                    'reprojection_error': float(reprojection_error) if reference_landmarks is not None else 0.0
                }
            else:
                # Basit yüz oranı kontrolü
                aspect_ratio = w / h
                ideal_ratio = 0.75  # İdeal yüz oranı
                ratio_score = 1 - abs(aspect_ratio - ideal_ratio) / ideal_ratio
                
                return {
                    'alignment_score': max(0, float(ratio_score)),
                    'landmark_count': 0,
                    'aspect_ratio': float(aspect_ratio)
                }
                
        except Exception as e:
            print(f"Face alignment hesaplama hatası: {e}")
            return {'alignment_score': 0.0, 'landmark_count': 0, 'error': str(e)}
    
    def _procrustes_analysis(self, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """Procrustes analizi ile landmark alignment"""
        # Merkez noktalarını hesapla
        center1 = np.mean(points1, axis=0)
        center2 = np.mean(points2, axis=0)
        
        # Merkez noktalarından uzaklıkları hesapla
        centered1 = points1 - center1
        centered2 = points2 - center2
        
        # Ölçeklendirme
        scale1 = np.sqrt(np.sum(centered1 ** 2))
        scale2 = np.sqrt(np.sum(centered2 ** 2))
        
        if scale1 > 0 and scale2 > 0:
            centered1 = centered1 / scale1
            centered2 = centered2 / scale2
        
        # Rotasyon matrisi hesapla
        H = centered1.T @ centered2
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Transformasyon uygula
        aligned = (centered1 @ R.T) * scale2 + center2
        
        return aligned
    
    def _calculate_landmark_quality(self, landmarks: np.ndarray) -> float:
        """Landmark kalite skoru"""
        try:
            # Yüz konturu (0-16)
            face_contour = landmarks[0:17]
            
            # Gözler (36-47)
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            
            # Burun (27-35)
            nose = landmarks[27:35]
            
            # Ağız (48-67)
            mouth = landmarks[48:68]
            
            # Simetri kontrolü
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
            
            # Yüz genişliği
            face_width = np.max(face_contour[:, 0]) - np.min(face_contour[:, 0])
            
            # Göz mesafesi oranı (ideal: 0.3-0.4)
            eye_ratio = eye_distance / face_width
            
            # Kalite skoru
            quality_score = 1.0
            if eye_ratio < 0.25 or eye_ratio > 0.45:
                quality_score -= 0.3
            
            return max(0, quality_score)
            
        except Exception as e:
            return 0.0
    
    def calculate_color_delta_e(self, before: np.ndarray, after: np.ndarray) -> Dict[str, float]:
        """Color ΔE (CIELAB) hesaplama"""
        try:
            # BGR'den LAB'ye dönüştür
            before_lab = cv2.cvtColor(before, cv2.COLOR_BGR2LAB)
            after_lab = cv2.cvtColor(after, cv2.COLOR_BGR2LAB)
            
            # ΔE hesapla
            delta_e = np.sqrt(np.sum((before_lab - after_lab) ** 2, axis=2))
            
            return {
                'delta_e_mean': float(np.mean(delta_e)),
                'delta_e_std': float(np.std(delta_e)),
                'delta_e_max': float(np.max(delta_e)),
                'delta_e_95th': float(np.percentile(delta_e, 95))
            }
            
        except Exception as e:
            return {'delta_e_mean': 0.0, 'error': str(e)}
    
    def calculate_temporal_jitter(self, flow_vectors: List[np.ndarray]) -> Dict[str, float]:
        """Temporal jitter hesaplama"""
        try:
            if not flow_vectors:
                return {'temporal_jitter': 0.0, 'flow_std': 0.0}
            
            # Flow vektörlerinin standart sapması
            flow_array = np.array(flow_vectors)
            flow_std = np.std(flow_array, axis=0)
            
            # Temporal jitter (genel hareket tutarsızlığı)
            temporal_jitter = np.mean(flow_std)
            
            return {
                'temporal_jitter': float(temporal_jitter),
                'flow_std_x': float(flow_std[0]),
                'flow_std_y': float(flow_std[1]),
                'flow_consistency': float(1 / (1 + temporal_jitter))  # 0-1 arası
            }
            
        except Exception as e:
            return {'temporal_jitter': 0.0, 'error': str(e)}
    
    def calculate_lip_sync_score(self, audio_path: str, video_path: str) -> Dict[str, float]:
        """Lip-sync skoru (LSE-C / LSE-D)"""
        try:
            # Basit lip-sync metrik (gerçek implementasyon için daha gelişmiş model gerekli)
            # Bu örnekte sadece placeholder değerler
            
            # Video'dan yüz tespiti
            cap = cv2.VideoCapture(video_path)
            face_detections = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Yüz tespiti
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                face_detections.append(len(faces) > 0)
            
            cap.release()
            
            # Basit lip-sync skoru (yüz tespiti oranına dayalı)
            face_detection_rate = sum(face_detections) / len(face_detections)
            
            return {
                'lse_c': float(face_detection_rate),  # Placeholder
                'lse_d': float(1 - face_detection_rate),  # Placeholder
                'face_detection_rate': float(face_detection_rate),
                'total_frames': len(face_detections)
            }
            
        except Exception as e:
            return {'lse_c': 0.0, 'lse_d': 1.0, 'error': str(e)}
    
    def calculate_psnr(self, before: np.ndarray, after: np.ndarray) -> float:
        """Peak Signal-to-Noise Ratio"""
        try:
            mse = np.mean((before - after) ** 2)
            if mse == 0:
                return float('inf')
            
            max_pixel = 255.0
            psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
            return float(psnr)
            
        except Exception as e:
            return 0.0
    
    def calculate_ssim(self, before: np.ndarray, after: np.ndarray) -> float:
        """Structural Similarity Index"""
        try:
            # Basit SSIM implementasyonu
            mu1 = np.mean(before)
            mu2 = np.mean(after)
            
            sigma1 = np.var(before)
            sigma2 = np.var(after)
            sigma12 = np.mean((before - mu1) * (after - mu2))
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
            
            return float(ssim)
            
        except Exception as e:
            return 0.0
    
    def calculate_comprehensive_metrics(self, before_path: str, after_path: str, 
                                      audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Kapsamlı metrik hesaplama"""
        start_time = datetime.now()
        
        try:
            # Görüntüleri yükle
            before = cv2.imread(before_path)
            after = cv2.imread(after_path)
            
            if before is None or after is None:
                return {'error': 'Görüntü yükleme hatası'}
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'before_path': before_path,
                'after_path': after_path
            }
            
            # Face alignment
            face_metrics = self.calculate_face_alignment_score(after)
            metrics.update(face_metrics)
            
            # Color metrics
            color_metrics = self.calculate_color_delta_e(before, after)
            metrics.update(color_metrics)
            
            # Image quality
            metrics['psnr'] = self.calculate_psnr(before, after)
            metrics['ssim'] = self.calculate_ssim(before, after)
            
            # Lip-sync (video varsa)
            if audio_path and Path(audio_path).exists():
                lip_sync_metrics = self.calculate_lip_sync_score(audio_path, after_path)
                metrics.update(lip_sync_metrics)
            
            # Genel kalite skoru
            quality_score = self._calculate_overall_quality_score(metrics)
            metrics['overall_quality_score'] = quality_score
            
            # İşlem süresi
            metrics['calculation_time'] = (datetime.now() - start_time).total_seconds()
            
            return metrics
            
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _calculate_overall_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Genel kalite skoru hesaplama"""
        try:
            # Ağırlıklı skor hesaplama
            weights = {
                'alignment_score': 0.3,
                'ssim': 0.25,
                'psnr': 0.2,
                'delta_e_mean': 0.15,
                'face_detection_rate': 0.1
            }
            
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    value = metrics[metric]
                    
                    # Normalize et
                    if metric == 'delta_e_mean':
                        # ΔE için ters oran (düşük ΔE = yüksek kalite)
                        normalized = max(0, 1 - (value / 50))
                    elif metric == 'psnr':
                        # PSNR için normalize (30+ = yüksek kalite)
                        normalized = min(1, value / 30)
                    else:
                        normalized = min(1, max(0, value))
                    
                    score += normalized * weight
                    total_weight += weight
            
            return float(score / total_weight) if total_weight > 0 else 0.0
            
        except Exception as e:
            return 0.0
    
    def save_metrics(self, metrics: Dict[str, Any], job_id: int) -> str:
        """Metrikleri dosyaya kaydet"""
        try:
            metrics_dir = Path("storage/logs/metrics")
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            metrics_file = metrics_dir / f"job_{job_id}_metrics.json"
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            return str(metrics_file)
            
        except Exception as e:
            print(f"Metrik kaydetme hatası: {e}")
            return ""

# Global instance
_metrics_calculator = None

def get_metrics_calculator() -> ObjectiveMetrics:
    """Global metrics calculator instance"""
    global _metrics_calculator
    if _metrics_calculator is None:
        _metrics_calculator = ObjectiveMetrics()
    return _metrics_calculator
