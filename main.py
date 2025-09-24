#!/usr/bin/env python3
"""
Fotoğraf ve Video İşleme Sistemi - Ana Kontrol Scripti
Tüm özellikleri tek yerden yönetir
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Yerel modülleri import et
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from face_enhancer import FaceEnhancer
from face_swapper import FaceSwapper
from video_processor import VideoProcessor
from talking_head import TalkingHeadGenerator

class PhotoVideoProcessor:
    def __init__(self):
        """Ana işlemci sınıfı"""
        self.version = "1.0.0"
        self.setup_directories()
        
    def setup_directories(self):
        """Gerekli klasörleri oluştur"""
        directories = [
            "data/input",
            "data/output", 
            "data/models",
            "outputs/faces",
            "outputs/enhanced",
            "outputs/swap",
            "outputs/videos",
            "outputs/talking_heads",
            "work/frames",
            "work/temp"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def print_banner(self):
        """Sistem banner'ını yazdır"""
        banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                Fotoğraf ve Video İşleme Sistemi              ║
║                        Versiyon {self.version}                        ║
╠══════════════════════════════════════════════════════════════╣
║  🎯 Yüz Tespiti ve Tanıma                                   ║
║  🔧 Yüz İyileştirme ve Düzenleme                            ║
║  🔄 Yüz Değiştirme (Face Swap)                              ║
║  🎬 Video İşleme ve Yüz Yerleştirme                         ║
║  🗣️  Konuşan Kafa Üretimi (Talking Head)                    ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def detect_faces(self, image_path: str, method: str = 'opencv', save_faces: bool = False):
        """Yüz tespiti yap"""
        print(f"\n🔍 Yüz Tespiti Başlıyor...")
        detector = FaceDetector(method=method)
        result = detector.detect_faces(image_path)
        
        if result is None:
            print("❌ Yüz tespiti başarısız!")
            return
        
        faces_data, original_image = result
        
        print(f"✅ {faces_data['faces_count']} yüz tespit edildi")
        
        if save_faces:
            saved_files = detector.save_faces(faces_data, original_image)
            print(f"💾 {len(saved_files)} yüz dosyası kaydedildi")
        
        # Sonuç görüntüsünü kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/faces/detection_{timestamp}.jpg"
        result_image = detector.draw_detections(original_image, faces_data)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        import cv2
        cv2.imwrite(output_path, result_image)
        print(f"🖼️  Sonuç görüntüsü: {output_path}")
    
    def recognize_faces(self, image1: str, image2: str, method: str = 'face_recognition'):
        """Yüz tanıma yap"""
        print(f"\n👤 Yüz Tanıma Başlıyor...")
        recognizer = FaceRecognizer(method=method)
        result = recognizer.compare_faces(image1, image2)
        
        if result['success']:
            print(f"📊 Benzerlik: {result['similarity']:.3f}")
            print(f"🎯 Aynı kişi mi: {'✅ Evet' if result['is_same_person'] else '❌ Hayır'}")
            print(f"🔒 Güven seviyesi: {result.get('confidence', 'N/A')}")
        else:
            print(f"❌ Hata: {result['error']}")
    
    def enhance_face(self, image_path: str, method: str = 'gfpgan'):
        """Yüz iyileştirme yap"""
        print(f"\n🔧 Yüz İyileştirme Başlıyor...")
        enhancer = FaceEnhancer(method=method)
        result = enhancer.enhance_face(image_path)
        
        if result['success']:
            print(f"✅ İyileştirme tamamlandı!")
            print(f"📊 Kalite skoru: {result['quality_score']:.3f}")
            print(f"📁 Çıktı: {result['output_path']}")
        else:
            print(f"❌ Hata: {result['error']}")
    
    def swap_faces(self, source_path: str, target_path: str, method: str = 'insightface'):
        """Yüz değiştirme yap"""
        print(f"\n🔄 Yüz Değiştirme Başlıyor...")
        swapper = FaceSwapper(method=method)
        result = swapper.swap_faces(source_path, target_path)
        
        if result['success']:
            print(f"✅ Yüz değiştirme tamamlandı!")
            print(f"🔄 Değiştirilen yüz sayısı: {result['swapped_faces']}")
            print(f"📁 Çıktı: {result['output_path']}")
        else:
            print(f"❌ Hata: {result['error']}")
    
    def process_video(self, video_path: str, source_face_path: str, method: str = 'insightface'):
        """Video işleme yap"""
        print(f"\n🎬 Video İşleme Başlıyor...")
        processor = VideoProcessor(method=method)
        result = processor.swap_face_in_video(video_path, source_face_path)
        
        if result['success']:
            print(f"✅ Video işleme tamamlandı!")
            print(f"🎬 İşlenen frame: {result['processed_frames']}")
            print(f"📁 Çıktı: {result['output_path']}")
        else:
            print(f"❌ Hata: {result['error']}")
    
    def generate_talking_head(self, image_path: str, audio_path: str, method: str = 'wav2lip'):
        """Konuşan kafa üret"""
        print(f"\n🗣️  Konuşan Kafa Üretimi Başlıyor...")
        generator = TalkingHeadGenerator(method=method)
        result = generator.generate_talking_head(image_path, audio_path)
        
        if result['success']:
            print(f"✅ Konuşan kafa üretimi tamamlandı!")
            print(f"🎬 Yöntem: {result['method']}")
            print(f"📁 Çıktı: {result['output_path']}")
        else:
            print(f"❌ Hata: {result['error']}")
    
    def interactive_mode(self):
        """Etkileşimli mod"""
        self.print_banner()
        
        while True:
            print(f"\n{'='*60}")
            print("🎯 Ana Menü - Ne yapmak istiyorsunuz?")
            print("1. 🔍 Yüz Tespiti")
            print("2. 👤 Yüz Tanıma")
            print("3. 🔧 Yüz İyileştirme")
            print("4. 🔄 Yüz Değiştirme")
            print("5. 🎬 Video İşleme")
            print("6. 🗣️  Konuşan Kafa Üretimi")
            print("7. 📊 Sistem Durumu")
            print("0. 🚪 Çıkış")
            print(f"{'='*60}")
            
            try:
                choice = input("Seçiminizi yapın (0-7): ").strip()
                
                if choice == '0':
                    print("👋 Görüşürüz!")
                    break
                elif choice == '1':
                    self.handle_face_detection()
                elif choice == '2':
                    self.handle_face_recognition()
                elif choice == '3':
                    self.handle_face_enhancement()
                elif choice == '4':
                    self.handle_face_swapping()
                elif choice == '5':
                    self.handle_video_processing()
                elif choice == '6':
                    self.handle_talking_head()
                elif choice == '7':
                    self.show_system_status()
                else:
                    print("❌ Geçersiz seçim! Lütfen 0-7 arası bir sayı girin.")
                    
            except KeyboardInterrupt:
                print("\n👋 Görüşürüz!")
                break
            except Exception as e:
                print(f"❌ Hata: {e}")
    
    def handle_face_detection(self):
        """Yüz tespiti işlemini yönet"""
        print(f"\n🔍 Yüz Tespiti")
        image_path = input("Görüntü dosyası yolu: ").strip()
        
        if not os.path.exists(image_path):
            print("❌ Dosya bulunamadı!")
            return
        
        print("Yöntem seçin:")
        print("1. OpenCV (Hızlı)")
        print("2. MediaPipe (Orta)")
        print("3. InsightFace (En iyi)")
        
        method_choice = input("Seçim (1-3): ").strip()
        method_map = {'1': 'opencv', '2': 'mediapipe', '3': 'insightface'}
        method = method_map.get(method_choice, 'opencv')
        
        save_faces = input("Yüzleri ayrı dosyalar olarak kaydet? (y/n): ").strip().lower() == 'y'
        
        self.detect_faces(image_path, method, save_faces)
    
    def handle_face_recognition(self):
        """Yüz tanıma işlemini yönet"""
        print(f"\n👤 Yüz Tanıma")
        image1 = input("İlk görüntü yolu: ").strip()
        image2 = input("İkinci görüntü yolu: ").strip()
        
        if not os.path.exists(image1) or not os.path.exists(image2):
            print("❌ Dosyalardan biri bulunamadı!")
            return
        
        self.recognize_faces(image1, image2)
    
    def handle_face_enhancement(self):
        """Yüz iyileştirme işlemini yönet"""
        print(f"\n🔧 Yüz İyileştirme")
        image_path = input("İyileştirilecek görüntü yolu: ").strip()
        
        if not os.path.exists(image_path):
            print("❌ Dosya bulunamadı!")
            return
        
        print("Yöntem seçin:")
        print("1. OpenCV (Hızlı)")
        print("2. GFPGAN (En iyi)")
        print("3. Real-ESRGAN (Orta)")
        
        method_choice = input("Seçim (1-3): ").strip()
        method_map = {'1': 'opencv', '2': 'gfpgan', '3': 'real_esrgan'}
        method = method_map.get(method_choice, 'gfpgan')
        
        self.enhance_face(image_path, method)
    
    def handle_face_swapping(self):
        """Yüz değiştirme işlemini yönet"""
        print(f"\n🔄 Yüz Değiştirme")
        source_path = input("Kaynak yüz görüntüsü: ").strip()
        target_path = input("Hedef görüntü: ").strip()
        
        if not os.path.exists(source_path) or not os.path.exists(target_path):
            print("❌ Dosyalardan biri bulunamadı!")
            return
        
        print("Yöntem seçin:")
        print("1. OpenCV (Hızlı)")
        print("2. InsightFace (En iyi)")
        
        method_choice = input("Seçim (1-2): ").strip()
        method_map = {'1': 'opencv', '2': 'insightface'}
        method = method_map.get(method_choice, 'insightface')
        
        self.swap_faces(source_path, target_path, method)
    
    def handle_video_processing(self):
        """Video işleme işlemini yönet"""
        print(f"\n🎬 Video İşleme")
        video_path = input("Video dosyası yolu: ").strip()
        source_face_path = input("Kaynak yüz görüntüsü: ").strip()
        
        if not os.path.exists(video_path) or not os.path.exists(source_face_path):
            print("❌ Dosyalardan biri bulunamadı!")
            return
        
        self.process_video(video_path, source_face_path)
    
    def handle_talking_head(self):
        """Konuşan kafa işlemini yönet"""
        print(f"\n🗣️  Konuşan Kafa Üretimi")
        image_path = input("Görüntü dosyası yolu: ").strip()
        audio_path = input("Ses dosyası yolu: ").strip()
        
        if not os.path.exists(image_path) or not os.path.exists(audio_path):
            print("❌ Dosyalardan biri bulunamadı!")
            return
        
        print("Yöntem seçin:")
        print("1. Wav2Lip (En iyi)")
        print("2. SadTalker (Alternatif)")
        print("3. OpenCV (Demo)")
        
        method_choice = input("Seçim (1-3): ").strip()
        method_map = {'1': 'wav2lip', '2': 'sadtalker', '3': 'opencv'}
        method = method_map.get(method_choice, 'wav2lip')
        
        self.generate_talking_head(image_path, audio_path, method)
    
    def show_system_status(self):
        """Sistem durumunu göster"""
        print(f"\n📊 Sistem Durumu")
        print(f"Versiyon: {self.version}")
        print(f"Python: {sys.version}")
        
        # Klasör durumları
        directories = [
            "data/input", "data/output", "outputs/faces", 
            "outputs/enhanced", "outputs/swap", "outputs/videos", 
            "outputs/talking_heads"
        ]
        
        print(f"\n📁 Klasör Durumları:")
        for directory in directories:
            path = Path(directory)
            if path.exists():
                file_count = len(list(path.glob("*")))
                print(f"  ✅ {directory}: {file_count} dosya")
            else:
                print(f"  ❌ {directory}: Bulunamadı")
        
        # Model durumları
        print(f"\n🤖 Model Durumları:")
        models = [
            ("models/wav2lip", "Wav2Lip"),
            ("models/sadtalker", "SadTalker")
        ]
        
        for model_path, model_name in models:
            path = Path(model_path)
            if path.exists():
                print(f"  ✅ {model_name}: Kurulu")
            else:
                print(f"  ⚠️  {model_name}: Kurulu değil (ilk kullanımda indirilecek)")

def main():
    parser = argparse.ArgumentParser(description='Fotoğraf ve Video İşleme Sistemi')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Etkileşimli modu başlat')
    parser.add_argument('--detect', help='Yüz tespiti (görüntü yolu)')
    parser.add_argument('--recognize', nargs=2, metavar=('IMAGE1', 'IMAGE2'),
                       help='Yüz tanıma (iki görüntü)')
    parser.add_argument('--enhance', help='Yüz iyileştirme (görüntü yolu)')
    parser.add_argument('--swap', nargs=2, metavar=('SOURCE', 'TARGET'),
                       help='Yüz değiştirme (kaynak, hedef)')
    parser.add_argument('--video', nargs=2, metavar=('VIDEO', 'FACE'),
                       help='Video işleme (video, yüz)')
    parser.add_argument('--talking', nargs=2, metavar=('IMAGE', 'AUDIO'),
                       help='Konuşan kafa (görüntü, ses)')
    
    args = parser.parse_args()
    
    # Ana işlemci oluştur
    processor = PhotoVideoProcessor()
    
    if args.interactive or len(sys.argv) == 1:
        # Etkileşimli mod
        processor.interactive_mode()
    
    elif args.detect:
        # Yüz tespiti
        processor.detect_faces(args.detect)
    
    elif args.recognize:
        # Yüz tanıma
        processor.recognize_faces(args.recognize[0], args.recognize[1])
    
    elif args.enhance:
        # Yüz iyileştirme
        processor.enhance_face(args.enhance)
    
    elif args.swap:
        # Yüz değiştirme
        processor.swap_faces(args.swap[0], args.swap[1])
    
    elif args.video:
        # Video işleme
        processor.process_video(args.video[0], args.video[1])
    
    elif args.talking:
        # Konuşan kafa
        processor.generate_talking_head(args.talking[0], args.talking[1])
    
    else:
        # Yardım göster
        processor.print_banner()
        print("\nKullanım örnekleri:")
        print("  python main.py --interactive")
        print("  python main.py --detect photo.jpg")
        print("  python main.py --recognize face1.jpg face2.jpg")
        print("  python main.py --enhance photo.jpg")
        print("  python main.py --swap source.jpg target.jpg")
        print("  python main.py --video video.mp4 face.jpg")
        print("  python main.py --talking photo.jpg audio.wav")

if __name__ == "__main__":
    main()
