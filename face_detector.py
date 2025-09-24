#!/usr/bin/env python3
"""
Gelişmiş Yüz Tespiti ve Analiz Sistemi
OpenCV, MediaPipe ve InsightFace kullanarak yüz tespiti yapar
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime

class FaceDetector:
    def __init__(self, method='opencv'):
        """
        Yüz tespit sistemi başlatıcı
        
        Args:
            method (str): 'opencv', 'mediapipe', 'insightface' seçenekleri
        """
        self.method = method
        self.setup_detector()
        
    def setup_detector(self):
        """Seçilen yönteme göre tespit sistemini kur"""
        if self.method == 'opencv':
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            print("✅ OpenCV yüz tespit sistemi hazır")
            
        elif self.method == 'mediapipe':
            try:
                import mediapipe as mp
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_drawing = mp.solutions.drawing_utils
                self.detector = self.mp_face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
                print("✅ MediaPipe yüz tespit sistemi hazır")
            except ImportError:
                print("❌ MediaPipe kurulu değil, OpenCV'ye geçiliyor")
                self.method = 'opencv'
                self.setup_detector()
                
        elif self.method == 'insightface':
            try:
                from insightface.app import FaceAnalysis
                self.detector = FaceAnalysis(name='buffalo_l')
                self.detector.prepare(ctx_id=0, det_size=(640, 640))
                print("✅ InsightFace yüz tespit sistemi hazır")
            except ImportError:
                print("❌ InsightFace kurulu değil, OpenCV'ye geçiliyor")
                self.method = 'opencv'
                self.setup_detector()
    
    def detect_faces_opencv(self, image):
        """OpenCV ile yüz tespiti"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 4)
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': 1.0,  # OpenCV confidence vermiyor
                'landmarks': None
            })
        return results
    
    def detect_faces_mediapipe(self, image):
        """MediaPipe ile yüz tespiti"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_image)
        
        face_data = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                face_data.append({
                    'bbox': [x, y, width, height],
                    'confidence': detection.score[0],
                    'landmarks': None
                })
        return face_data
    
    def detect_faces_insightface(self, image):
        """InsightFace ile yüz tespiti"""
        faces = self.detector.get(image)
        
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            results.append({
                'bbox': [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
                'confidence': face.det_score,
                'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None
            })
        return results
    
    def detect_faces(self, image_path):
        """Ana yüz tespit fonksiyonu"""
        if not os.path.exists(image_path):
            print(f"❌ Dosya bulunamadı: {image_path}")
            return None
        
        # Görüntüyü yükle
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Görüntü yüklenemedi: {image_path}")
            return None
        
        print(f"🔍 {self.method.upper()} ile yüz tespiti yapılıyor...")
        
        # Yönteme göre tespit yap
        if self.method == 'opencv':
            faces = self.detect_faces_opencv(image)
        elif self.method == 'mediapipe':
            faces = self.detect_faces_mediapipe(image)
        elif self.method == 'insightface':
            faces = self.detect_faces_insightface(image)
        
        print(f"✅ {len(faces)} yüz tespit edildi")
        
        # Sonuçları kaydet
        result = {
            'image_path': image_path,
            'method': self.method,
            'timestamp': datetime.now().isoformat(),
            'faces_count': len(faces),
            'faces': faces
        }
        
        return result, image
    
    def save_faces(self, faces_data, original_image, output_dir="outputs/faces"):
        """Tespit edilen yüzleri ayrı dosyalar olarak kaydet"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        saved_faces = []
        for i, face in enumerate(faces_data['faces']):
            x, y, w, h = face['bbox']
            
            # Yüzü kırp
            face_crop = original_image[y:y+h, x:x+w]
            
            # Dosya adı oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_{i}_{timestamp}.jpg"
            filepath = Path(output_dir) / filename
            
            # Kaydet
            cv2.imwrite(str(filepath), face_crop)
            saved_faces.append(str(filepath))
            
            print(f"💾 Yüz {i+1} kaydedildi: {filepath}")
        
        return saved_faces
    
    def draw_detections(self, image, faces_data):
        """Tespit edilen yüzleri görüntü üzerinde işaretle"""
        result_image = image.copy()
        
        for i, face in enumerate(faces_data['faces']):
            x, y, w, h = face['bbox']
            confidence = face['confidence']
            
            # Dikdörtgen çiz
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
            
            # Etiket ekle
            label = f"Yüz {i+1}: {confidence:.2f}"
            cv2.putText(result_image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Landmark'ları çiz (varsa)
            if face['landmarks']:
                for landmark in face['landmarks']:
                    cv2.circle(result_image, (int(landmark[0]), int(landmark[1])), 
                              2, (255, 0, 0), -1)
        
        return result_image

def main():
    parser = argparse.ArgumentParser(description='Gelişmiş Yüz Tespiti')
    parser.add_argument('image_path', help='Tespit edilecek görüntü yolu')
    parser.add_argument('--method', choices=['opencv', 'mediapipe', 'insightface'], 
                       default='opencv', help='Tespit yöntemi')
    parser.add_argument('--save-faces', action='store_true', 
                       help='Tespit edilen yüzleri ayrı dosyalar olarak kaydet')
    parser.add_argument('--output', default='outputs/detection_result.jpg',
                       help='Sonuç görüntüsü çıktı yolu')
    
    args = parser.parse_args()
    
    # Tespit sistemi oluştur
    detector = FaceDetector(method=args.method)
    
    # Yüz tespiti yap
    result = detector.detect_faces(args.image_path)
    if result is None:
        return
    
    faces_data, original_image = result
    
    # Sonuçları göster
    print(f"\n📊 Tespit Sonuçları:")
    print(f"   Yöntem: {faces_data['method']}")
    print(f"   Tespit edilen yüz sayısı: {faces_data['faces_count']}")
    
    for i, face in enumerate(faces_data['faces']):
        print(f"   Yüz {i+1}: Güven skoru {face['confidence']:.3f}")
    
    # Yüzleri kaydet
    if args.save_faces:
        saved_files = detector.save_faces(faces_data, original_image)
        print(f"\n💾 {len(saved_files)} yüz dosyası kaydedildi")
    
    # Sonuç görüntüsünü oluştur ve kaydet
    result_image = detector.draw_detections(original_image, faces_data)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, result_image)
    print(f"🖼️  Sonuç görüntüsü kaydedildi: {args.output}")
    
    # JSON raporu kaydet
    json_path = args.output.replace('.jpg', '_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(faces_data, f, indent=2, ensure_ascii=False)
    print(f"📄 Detaylı rapor kaydedildi: {json_path}")

if __name__ == "__main__":
    main()
