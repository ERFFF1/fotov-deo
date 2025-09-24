#!/usr/bin/env python3
"""
Ethics & Security Manager
Etik ve güvenlik katmanı - rıza yönetimi, filigran, içerik etiketleme
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import hashlib
import json

class ConsentManager:
    """Rıza yönetim sistemi"""
    
    def __init__(self):
        self.consent_types = {
            'consented': {
                'name': 'Açık Rıza',
                'description': 'Kişi açık rıza vermiş',
                'color': '#27ae60',
                'requires_evidence': True
            },
            'demo': {
                'name': 'Demo/Kurgu',
                'description': 'Eğitim veya demo amaçlı',
                'color': '#f39c12',
                'requires_evidence': False
            },
            'unknown': {
                'name': 'Bilinmeyen',
                'description': 'Rıza durumu belirsiz',
                'color': '#e74c3c',
                'requires_evidence': True
            }
        }
    
    def validate_consent(self, consent_tag: str, evidence_path: str = None) -> Dict[str, Any]:
        """Rıza durumunu doğrula"""
        if consent_tag not in self.consent_types:
            return {
                'valid': False,
                'error': f'Geçersiz rıza etiketi: {consent_tag}',
                'allowed_operations': []
            }
        
        consent_info = self.consent_types[consent_tag]
        
        # Kanıt gereksinimi kontrolü
        if consent_info['requires_evidence'] and not evidence_path:
            return {
                'valid': False,
                'error': f'{consent_info["name"]} için kanıt dosyası gerekli',
                'allowed_operations': []
            }
        
        # İzin verilen işlemler
        allowed_operations = self._get_allowed_operations(consent_tag)
        
        return {
            'valid': True,
            'consent_info': consent_info,
            'allowed_operations': allowed_operations,
            'evidence_required': consent_info['requires_evidence']
        }
    
    def _get_allowed_operations(self, consent_tag: str) -> List[str]:
        """Rıza türüne göre izin verilen işlemleri al"""
        operation_matrix = {
            'consented': [
                'face_detect', 'face_recognize', 'face_enhance', 'face_swap',
                'video_swap', 'talking_head', 'color_grade', 'bg_replace',
                'body_segment', 'outfit_replace', 'audio_enhance'
            ],
            'demo': [
                'face_detect', 'face_enhance', 'color_grade', 'audio_enhance'
            ],
            'unknown': [
                'face_detect', 'color_grade', 'audio_enhance'
            ]
        }
        
        return operation_matrix.get(consent_tag, [])
    
    def generate_consent_warning(self, consent_tag: str) -> str:
        """Rıza uyarısı oluştur"""
        consent_info = self.consent_types.get(consent_tag, {})
        
        warnings = {
            'consented': '✅ Bu içerik için açık rıza alınmıştır.',
            'demo': '⚠️ Bu içerik demo/kurgu amaçlıdır. Ticari kullanım yasaktır.',
            'unknown': '🚨 Rıza durumu belirsiz! Yasal sorumluluk kullanıcıya aittir.'
        }
        
        return warnings.get(consent_tag, '❓ Bilinmeyen rıza durumu')
    
    def get_watermark_policy(self, consent_tag: str) -> Dict[str, Any]:
        """Filigran politikası döndür"""
        policies = {
            "consented": {
                "required": False,
                "opacity": 0.15,
                "size": "small",
                "position": "bottom-right",
                "text": "AI-Generated",
                "description": "İzinli içerik - minimal filigran"
            },
            "demo": {
                "required": True,
                "opacity": 0.20,
                "size": "medium",
                "position": "bottom-right",
                "text": "AI-Generated Demo",
                "description": "Demo içerik - orta seviye filigran"
            },
            "unknown": {
                "required": True,
                "opacity": 0.25,
                "size": "large",
                "position": "bottom-right",
                "text": "AI-Generated Unknown",
                "description": "Bilinmeyen içerik - büyük filigran"
            }
        }
        
        return policies.get(consent_tag, policies["unknown"])
    
    def validate_watermark_requirement(self, consent_tag: str) -> bool:
        """Filigran gerekliliğini kontrol et"""
        policy = self.get_watermark_policy(consent_tag)
        return policy["required"]
    
    def get_watermark_statistics(self) -> Dict[str, Any]:
        """Filigran istatistiklerini döndür"""
        try:
            log_file = Path("storage/logs/watermark_applications.jsonl")
            if not log_file.exists():
                return {"total_applications": 0, "by_consent": {}}
            
            applications = []
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        applications.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            
            # İstatistikleri hesapla
            total = len(applications)
            by_consent = {}
            
            for app in applications:
                consent = app.get("consent_tag", "unknown")
                by_consent[consent] = by_consent.get(consent, 0) + 1
            
            return {
                "total_applications": total,
                "by_consent": by_consent,
                "recent_applications": applications[-10:] if len(applications) > 10 else applications
            }
            
        except Exception as e:
            print(f"Filigran istatistik hatası: {e}")
            return {"total_applications": 0, "by_consent": {}}

class WatermarkManager:
    """Filigran yönetim sistemi"""
    
    def __init__(self):
        self.watermark_templates = {
            'ai_generated': {
                'text': 'AI-Generated',
                'color': (255, 0, 0),  # Kırmızı
                'opacity': 0.7,
                'position': 'bottom_right'
            },
            'demo_content': {
                'text': 'DEMO CONTENT',
                'color': (255, 165, 0),  # Turuncu
                'opacity': 0.8,
                'position': 'top_left'
            },
            'consented': {
                'text': 'CONSENTED',
                'color': (0, 255, 0),  # Yeşil
                'opacity': 0.6,
                'position': 'bottom_left'
            }
        }
    
    def add_watermark(self, image_path: str, consent_tag: str, output_path: str = None) -> str:
        """Görüntüye filigran ekle"""
        try:
            # Görüntüyü yükle
            image = Image.open(image_path)
            
            # Filigran türünü belirle
            watermark_type = self._get_watermark_type(consent_tag)
            watermark_config = self.watermark_templates[watermark_type]
            
            # Filigran ekle
            watermarked_image = self._apply_watermark(image, watermark_config)
            
            # Çıktı dosya yolu
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"storage/outputs/watermarked_{timestamp}.png"
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            watermarked_image.save(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Filigran ekleme hatası: {e}")
            return image_path  # Hata durumunda orijinal dosyayı döndür
    
    def _get_watermark_type(self, consent_tag: str) -> str:
        """Rıza türüne göre filigran türünü belirle"""
        if consent_tag == 'consented':
            return 'consented'
        elif consent_tag == 'demo':
            return 'demo_content'
        else:
            return 'ai_generated'
    
    def _apply_watermark(self, image: Image.Image, config: Dict[str, Any]) -> Image.Image:
        """Robust filigranı görüntüye uygula"""
        # Şeffaflık için alpha kanalı ekle
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Filigran katmanı oluştur
        watermark = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Font boyutunu hesapla
        font_size = max(20, min(image.size) // 20)
        
        try:
            # Sistem fontunu kullan
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            # Varsayılan font
            font = ImageFont.load_default()
        
        # Metin boyutunu hesapla
        bbox = draw.textbbox((0, 0), config['text'], font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Pozisyonu hesapla
        position = self._calculate_position(image.size, (text_width, text_height), config['position'])
        
        # Köşe jitter (2-3 px)
        import random
        jitter_x = random.randint(-3, 4)
        jitter_y = random.randint(-3, 4)
        position = (position[0] + jitter_x, position[1] + jitter_y)
        
        # Opaklık ayarı (consent türüne göre)
        if 'unknown' in config['text'].lower():
            opacity = 0.25  # %25 opaklık
        elif 'demo' in config['text'].lower():
            opacity = 0.20  # %20 opaklık
        else:  # consented
            opacity = 0.15  # %15 opaklık
        
        # Renk ve şeffaflık
        color = (*config['color'], int(255 * opacity))
        
        # Metni çiz
        draw.text(position, config['text'], font=font, fill=color)
        
        # Görüntüyü birleştir
        result = Image.alpha_composite(image, watermark)
        
        # EXIF metadata ekle
        result = self._add_exif_metadata(result, config['text'])
        
        return result.convert('RGB')
    
    def _add_exif_metadata(self, image: Image.Image, watermark_text: str) -> Image.Image:
        """EXIF metadata ekle"""
        try:
            # EXIF verilerini al
            exif = image.getexif()
            
            # AI-Generated etiketi
            exif[0x9286] = "AI-Generated:true"  # UserComment tag
            
            # Filigran bilgisi
            exif[0x010E] = f"Watermark: {watermark_text}"  # ImageDescription tag
            
            # Tarih
            exif[0x0132] = datetime.now().strftime("%Y:%m:%d %H:%M:%S")  # DateTime tag
            
            # Yeni görüntü oluştur
            result = Image.new(image.mode, image.size)
            result.putdata(list(image.getdata()))
            result.save = lambda *args, **kwargs: image.save(*args, **kwargs, exif=exif)
            
            return result
            
        except Exception as e:
            print(f"EXIF metadata ekleme hatası: {e}")
            return image
    
    def _calculate_position(self, image_size: Tuple[int, int], text_size: Tuple[int, int], position: str) -> Tuple[int, int]:
        """Filigran pozisyonunu hesapla"""
        img_width, img_height = image_size
        text_width, text_height = text_size
        
        margin = 20
        
        positions = {
            'top_left': (margin, margin),
            'top_right': (img_width - text_width - margin, margin),
            'bottom_left': (margin, img_height - text_height - margin),
            'bottom_right': (img_width - text_width - margin, img_height - text_height - margin),
            'center': ((img_width - text_width) // 2, (img_height - text_height) // 2)
        }
        
        return positions.get(position, positions['bottom_right'])

class ContentAnalyzer:
    """İçerik analiz sistemi"""
    
    def __init__(self):
        self.sensitive_keywords = [
            'nude', 'naked', 'explicit', 'adult', 'porn', 'sex',
            'violence', 'blood', 'weapon', 'gun', 'knife'
        ]
    
    def analyze_content(self, image_path: str) -> Dict[str, Any]:
        """İçeriği analiz et"""
        try:
            # Görüntüyü yükle
            image = cv2.imread(image_path)
            if image is None:
                return {'safe': False, 'reason': 'Görüntü yüklenemedi'}
            
            # Temel analiz
            analysis = {
                'safe': True,
                'confidence': 1.0,
                'flags': [],
                'metadata': self._extract_metadata(image)
            }
            
            # Yüz tespiti
            faces = self._detect_faces(image)
            analysis['face_count'] = len(faces)
            
            # Hassas içerik kontrolü (basit implementasyon)
            if self._check_sensitive_content(image):
                analysis['safe'] = False
                analysis['flags'].append('sensitive_content')
                analysis['confidence'] = 0.3
            
            return analysis
            
        except Exception as e:
            return {'safe': False, 'reason': f'Analiz hatası: {str(e)}'}
    
    def _extract_metadata(self, image: np.ndarray) -> Dict[str, Any]:
        """Görüntü metadata'sını çıkar"""
        height, width = image.shape[:2]
        
        return {
            'dimensions': {'width': width, 'height': height},
            'channels': image.shape[2] if len(image.shape) > 2 else 1,
            'aspect_ratio': width / height,
            'total_pixels': width * height
        }
    
    def _detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Yüz tespiti yap"""
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            return [{'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)} 
                   for (x, y, w, h) in faces]
            
        except Exception as e:
            print(f"Yüz tespiti hatası: {e}")
            return []
    
    def _check_sensitive_content(self, image: np.ndarray) -> bool:
        """Hassas içerik kontrolü (basit implementasyon)"""
        # Bu basit bir implementasyon - gerçek uygulamada daha gelişmiş
        # AI modelleri kullanılmalı
        
        # Renk analizi (çok basit)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Ten rengi tespiti
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.sum(skin_mask > 0) / (image.shape[0] * image.shape[1])
        
        # Eğer çok fazla ten rengi varsa (basit kontrol)
        if skin_ratio > 0.7:
            return True
        
        return False

class AuditLogger:
    """Denetim log sistemi"""
    
    def __init__(self, log_dir: str = "storage/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log_operation(self, operation: str, user_id: str, consent_tag: str, 
                     input_files: List[str], output_files: List[str], 
                     metadata: Dict[str, Any] = None):
        """İşlem logunu kaydet"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'user_id': user_id,
            'consent_tag': consent_tag,
            'input_files': input_files,
            'output_files': output_files,
            'metadata': metadata or {},
            'session_id': self._generate_session_id()
        }
        
        # Log dosyası
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Log kaydetme hatası: {e}")
    
    def _generate_session_id(self) -> str:
        """Oturum ID'si oluştur"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    def get_audit_trail(self, user_id: str = None, operation: str = None, 
                       start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """Denetim izini al"""
        logs = []
        
        # Log dosyalarını tara
        for log_file in self.log_dir.glob("audit_*.jsonl"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        log_entry = json.loads(line.strip())
                        
                        # Filtreleme
                        if user_id and log_entry.get('user_id') != user_id:
                            continue
                        if operation and log_entry.get('operation') != operation:
                            continue
                        
                        # Tarih filtresi
                        log_time = datetime.fromisoformat(log_entry['timestamp'])
                        if start_date and log_time < start_date:
                            continue
                        if end_date and log_time > end_date:
                            continue
                        
                        logs.append(log_entry)
                        
            except Exception as e:
                print(f"Log okuma hatası {log_file}: {e}")
        
        return sorted(logs, key=lambda x: x['timestamp'], reverse=True)

# Global instances
consent_manager = ConsentManager()
watermark_manager = WatermarkManager()
content_analyzer = ContentAnalyzer()
audit_logger = AuditLogger()

def get_consent_manager() -> ConsentManager:
    """Consent manager'ı al"""
    return consent_manager

def get_watermark_manager() -> WatermarkManager:
    """Watermark manager'ı al"""
    return watermark_manager

def get_content_analyzer() -> ContentAnalyzer:
    """Content analyzer'ı al"""
    return content_analyzer

def get_audit_logger() -> AuditLogger:
    """Audit logger'ı al"""
    return audit_logger
