#!/usr/bin/env python3
"""
Gelişmiş Yüz Tanıma ve Karşılaştırma Sistemi
Face Recognition, InsightFace ve MediaPipe kullanarak yüz tanıma yapar
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
import argparse
import json
import pickle
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class FaceRecognizer:
    def __init__(self, method='face_recognition'):
        """
        Yüz tanıma sistemi başlatıcı
        
        Args:
            method (str): 'face_recognition', 'insightface', 'mediapipe' seçenekleri
        """
        self.method = method
        self.face_database = {}
        self.setup_recognizer()
        
    def setup_recognizer(self):
        """Seçilen yönteme göre tanıma sistemini kur"""
        if self.method == 'face_recognition':
            try:
                import face_recognition
                self.face_recognition = face_recognition
                print("✅ Face Recognition sistemi hazır")
            except ImportError:
                print("❌ Face Recognition kurulu değil, OpenCV'ye geçiliyor")
                self.method = 'opencv'
                self.setup_recognizer()
                
        elif self.method == 'insightface':
            try:
                from insightface.app import FaceAnalysis
                self.detector = FaceAnalysis(name='buffalo_l')
                self.detector.prepare(ctx_id=0, det_size=(640, 640))
                print("✅ InsightFace tanıma sistemi hazır")
            except ImportError:
                print("❌ InsightFace kurulu değil, OpenCV'ye geçiliyor")
                self.method = 'opencv'
                self.setup_recognizer()
                
        elif self.method == 'mediapipe':
            try:
                import mediapipe as mp
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_face_mesh = mp.solutions.face_mesh
                self.detector = self.mp_face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1, 
                    refine_landmarks=True, min_detection_confidence=0.5
                )
                print("✅ MediaPipe tanıma sistemi hazır")
            except ImportError:
                print("❌ MediaPipe kurulu değil, OpenCV'ye geçiliyor")
                self.method = 'opencv'
                self.setup_recognizer()
                
        elif self.method == 'opencv':
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                print("✅ OpenCV yüz tanıma sistemi hazır")
            except Exception as e:
                print(f"❌ OpenCV yüz tanıma sistemi kurulamadı: {e}")
                self.method = None
    
    def extract_face_encoding_face_recognition(self, image_path: str) -> Optional[np.ndarray]:
        """Face Recognition ile yüz encoding'i çıkar"""
        try:
            image = self.face_recognition.load_image_file(image_path)
            face_encodings = self.face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                return face_encodings[0]
            else:
                print(f"⚠️  {image_path} dosyasında yüz bulunamadı")
                return None
        except Exception as e:
            print(f"❌ Face Recognition hatası: {e}")
            return None
    
    def extract_face_encoding_insightface(self, image_path: str) -> Optional[np.ndarray]:
        """InsightFace ile yüz embedding'i çıkar"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Görüntü yüklenemedi: {image_path}")
                return None
            
            faces = self.detector.get(image)
            if len(faces) > 0:
                return faces[0].embedding
            else:
                print(f"⚠️  {image_path} dosyasında yüz bulunamadı")
                return None
        except Exception as e:
            print(f"❌ InsightFace hatası: {e}")
            return None
    
    def extract_face_encoding_mediapipe(self, image_path: str) -> Optional[np.ndarray]:
        """MediaPipe ile yüz özelliklerini çıkar"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Görüntü yüklenemedi: {image_path}")
                return None
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                # Landmark'ları numpy array'e çevir
                landmarks = []
                for face_landmarks in results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                return np.array(landmarks)
            else:
                print(f"⚠️  {image_path} dosyasında yüz bulunamadı")
                return None
        except Exception as e:
            print(f"❌ MediaPipe hatası: {e}")
            return None
    
    def extract_face_encoding(self, image_path: str) -> Optional[np.ndarray]:
        """Ana yüz encoding çıkarma fonksiyonu"""
        if not os.path.exists(image_path):
            print(f"❌ Dosya bulunamadı: {image_path}")
            return None
        
        if self.method == 'face_recognition':
            return self.extract_face_encoding_face_recognition(image_path)
        elif self.method == 'insightface':
            return self.extract_face_encoding_insightface(image_path)
        elif self.method == 'mediapipe':
            return self.extract_face_encoding_mediapipe(image_path)
    
    def add_face_to_database(self, name: str, image_path: str) -> bool:
        """Yüz veritabanına yeni kişi ekle"""
        encoding = self.extract_face_encoding(image_path)
        if encoding is not None:
            self.face_database[name] = {
                'encoding': encoding,
                'image_path': image_path,
                'added_date': datetime.now().isoformat()
            }
            print(f"✅ {name} veritabanına eklendi")
            return True
        else:
            print(f"❌ {name} eklenemedi - yüz bulunamadı")
            return False
    
    def compare_faces_face_recognition(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """Face Recognition ile yüz karşılaştırması"""
        distance = self.face_recognition.face_distance([encoding1], encoding2)[0]
        # Distance'ı similarity'ye çevir (0-1 arası, 1 = aynı)
        similarity = 1 - distance
        return similarity
    
    def compare_faces_insightface(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """InsightFace ile yüz karşılaştırması"""
        # Cosine similarity hesapla
        similarity = np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
        return float(similarity)
    
    def compare_faces_mediapipe(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """MediaPipe ile yüz karşılaştırması"""
        # Euclidean distance hesapla
        distance = np.linalg.norm(encoding1 - encoding2)
        # Distance'ı similarity'ye çevir
        similarity = 1 / (1 + distance)
        return float(similarity)
    
    def compare_faces(self, image1_path: str, image2_path: str) -> Dict:
        """İki görüntüdeki yüzleri karşılaştır"""
        print(f"🔍 Yüz karşılaştırması yapılıyor...")
        print(f"   Görüntü 1: {image1_path}")
        print(f"   Görüntü 2: {image2_path}")
        
        # Encoding'leri çıkar
        encoding1 = self.extract_face_encoding(image1_path)
        encoding2 = self.extract_face_encoding(image2_path)
        
        if encoding1 is None or encoding2 is None:
            return {
                'success': False,
                'error': 'Bir veya her iki görüntüde yüz bulunamadı',
                'similarity': 0.0
            }
        
        # Karşılaştırma yap
        if self.method == 'face_recognition':
            similarity = self.compare_faces_face_recognition(encoding1, encoding2)
        elif self.method == 'insightface':
            similarity = self.compare_faces_insightface(encoding1, encoding2)
        elif self.method == 'mediapipe':
            similarity = self.compare_faces_mediapipe(encoding1, encoding2)
        
        # Sonuç değerlendirmesi
        threshold = 0.6 if self.method == 'face_recognition' else 0.5
        is_same_person = similarity > threshold
        
        result = {
            'success': True,
            'image1_path': image1_path,
            'image2_path': image2_path,
            'method': self.method,
            'similarity': float(similarity),
            'threshold': threshold,
            'is_same_person': is_same_person,
            'confidence': 'Yüksek' if similarity > 0.8 else 'Orta' if similarity > 0.6 else 'Düşük',
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def recognize_face(self, image_path: str) -> Dict:
        """Veritabanındaki kişilerle yüz tanıma yap"""
        if not self.face_database:
            return {
                'success': False,
                'error': 'Veritabanı boş - önce kişi ekleyin',
                'recognized_person': None
            }
        
        print(f"🔍 Yüz tanıma yapılıyor: {image_path}")
        
        # Gelen görüntünün encoding'ini çıkar
        query_encoding = self.extract_face_encoding(image_path)
        if query_encoding is None:
            return {
                'success': False,
                'error': 'Görüntüde yüz bulunamadı',
                'recognized_person': None
            }
        
        # Veritabanındaki her kişiyle karşılaştır
        best_match = None
        best_similarity = 0.0
        
        for name, person_data in self.face_database.items():
            if self.method == 'face_recognition':
                similarity = self.compare_faces_face_recognition(query_encoding, person_data['encoding'])
            elif self.method == 'insightface':
                similarity = self.compare_faces_insightface(query_encoding, person_data['encoding'])
            elif self.method == 'mediapipe':
                similarity = self.compare_faces_mediapipe(query_encoding, person_data['encoding'])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        # Sonuç değerlendirmesi
        threshold = 0.6 if self.method == 'face_recognition' else 0.5
        recognized = best_match if best_similarity > threshold else None
        
        result = {
            'success': True,
            'image_path': image_path,
            'method': self.method,
            'recognized_person': recognized,
            'best_similarity': float(best_similarity),
            'threshold': threshold,
            'all_matches': {name: float(sim) for name, sim in 
                          [(name, self.compare_faces_face_recognition(query_encoding, data['encoding']) 
                           if self.method == 'face_recognition' else
                           self.compare_faces_insightface(query_encoding, data['encoding'])
                           if self.method == 'insightface' else
                           self.compare_faces_mediapipe(query_encoding, data['encoding']))
                           for name, data in self.face_database.items()]},
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def save_database(self, filepath: str = "data/face_database.pkl"):
        """Yüz veritabanını kaydet"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.face_database, f)
        print(f"💾 Veritabanı kaydedildi: {filepath}")
    
    def load_database(self, filepath: str = "data/face_database.pkl"):
        """Yüz veritabanını yükle"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.face_database = pickle.load(f)
            print(f"📂 Veritabanı yüklendi: {len(self.face_database)} kişi")
        else:
            print(f"⚠️  Veritabanı dosyası bulunamadı: {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Gelişmiş Yüz Tanıma Sistemi')
    parser.add_argument('--method', choices=['face_recognition', 'insightface', 'mediapipe'], 
                       default='face_recognition', help='Tanıma yöntemi')
    parser.add_argument('--compare', nargs=2, metavar=('IMAGE1', 'IMAGE2'),
                       help='İki görüntüyü karşılaştır')
    parser.add_argument('--recognize', metavar='IMAGE',
                       help='Görüntüdeki yüzü tanı')
    parser.add_argument('--add-person', nargs=2, metavar=('NAME', 'IMAGE'),
                       help='Kişiyi veritabanına ekle')
    parser.add_argument('--list-database', action='store_true',
                       help='Veritabanındaki kişileri listele')
    
    args = parser.parse_args()
    
    # Tanıma sistemi oluştur
    recognizer = FaceRecognizer(method=args.method)
    
    # Veritabanını yükle
    recognizer.load_database()
    
    if args.add_person:
        name, image_path = args.add_person
        recognizer.add_face_to_database(name, image_path)
        recognizer.save_database()
        
    elif args.compare:
        image1, image2 = args.compare
        result = recognizer.compare_faces(image1, image2)
        
        print(f"\n📊 Karşılaştırma Sonuçları:")
        print(f"   Benzerlik: {result['similarity']:.3f}")
        print(f"   Eşik değeri: {result['threshold']}")
        print(f"   Aynı kişi mi: {'✅ Evet' if result['is_same_person'] else '❌ Hayır'}")
        print(f"   Güven seviyesi: {result.get('confidence', 'N/A')}")
        
        # JSON raporu kaydet
        json_path = f"outputs/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"📄 Detaylı rapor: {json_path}")
        
    elif args.recognize:
        result = recognizer.recognize_face(args.recognize)
        
        if result['success']:
            print(f"\n👤 Tanıma Sonuçları:")
            if result['recognized_person']:
                print(f"   Tanınan kişi: {result['recognized_person']}")
                print(f"   Benzerlik: {result['best_similarity']:.3f}")
            else:
                print(f"   Tanınan kişi: Bilinmeyen")
                print(f"   En yüksek benzerlik: {result['best_similarity']:.3f}")
            
            print(f"\n📋 Tüm eşleşmeler:")
            for name, similarity in result['all_matches'].items():
                print(f"   {name}: {similarity:.3f}")
        else:
            print(f"❌ Hata: {result['error']}")
        
        # JSON raporu kaydet
        json_path = f"outputs/recognition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"📄 Detaylı rapor: {json_path}")
        
    elif args.list_database:
        print(f"\n👥 Veritabanındaki Kişiler ({len(recognizer.face_database)} kişi):")
        for name, data in recognizer.face_database.items():
            print(f"   {name}: {data['image_path']} (Eklenme: {data['added_date']})")
    
    else:
        print("❌ Lütfen bir işlem seçin. --help ile yardım alın.")

if __name__ == "__main__":
    main()
