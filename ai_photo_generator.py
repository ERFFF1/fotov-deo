#!/usr/bin/env python3
"""
AI Fotoğraf Üretim Sistemi
Stable Diffusion, ControlNet ve diğer AI modelleri ile fotoğraf üretimi
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

class AIPhotoGenerator:
    def __init__(self):
        """AI fotoğraf üretici başlatıcı"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_models()
        
    def setup_models(self):
        """AI modellerini kur"""
        try:
            from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline
            from diffusers import DPMSolverMultistepScheduler
            from controlnet_aux import CannyDetector, OpenposeDetector
            
            print(f"🤖 AI modelleri kuruluyor... (Cihaz: {self.device})")
            
            # Stable Diffusion pipeline
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Scheduler ayarla
            self.sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.sd_pipeline.scheduler.config
            )
            
            # ControlNet modelleri
            self.controlnet_canny = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
            
            self.controlnet_openpose = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose",
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
            
            # ControlNet pipeline'ları
            self.controlnet_pipeline_canny = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=self.controlnet_canny,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            self.controlnet_pipeline_openpose = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=self.controlnet_openpose,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # ControlNet aux modelleri
            self.canny_detector = CannyDetector()
            self.openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            
            print("✅ AI modelleri hazır")
            
        except ImportError as e:
            print(f"⚠️  AI modelleri kurulu değil: {e}")
            print("💡 Kurulum için: pip install diffusers transformers controlnet-aux")
            self.sd_pipeline = None
            self.controlnet_pipeline_canny = None
            self.controlnet_pipeline_openpose = None
    
    def generate_image(self, prompt: str, style: str = 'realistic', 
                      size: str = '512x512', steps: int = 20, 
                      guidance_scale: float = 7.5, seed: Optional[int] = None) -> Dict:
        """AI ile fotoğraf üret"""
        if self.sd_pipeline is None:
            return {
                'success': False,
                'error': 'AI modelleri kurulu değil',
                'output_path': None
            }
        
        try:
            # Boyutları parse et
            width, height = map(int, size.split('x'))
            
            # Stil prompt'unu ekle
            style_prompts = {
                'realistic': 'photorealistic, high quality, detailed',
                'anime': 'anime style, manga style, japanese animation',
                'oil_painting': 'oil painting, classical art, renaissance style',
                'watercolor': 'watercolor painting, soft colors, artistic',
                'sketch': 'pencil sketch, black and white, line art',
                'cyberpunk': 'cyberpunk, futuristic, neon lights, sci-fi',
                'fantasy': 'fantasy art, magical, mystical, ethereal',
                'portrait': 'professional portrait, studio lighting, high resolution'
            }
            
            enhanced_prompt = f"{prompt}, {style_prompts.get(style, '')}"
            
            # Seed ayarla
            if seed is not None:
                torch.manual_seed(seed)
            
            print(f"🎨 AI fotoğraf üretiliyor...")
            print(f"   Prompt: {enhanced_prompt}")
            print(f"   Boyut: {width}x{height}")
            print(f"   Adım: {steps}")
            
            # Görüntü üret
            with torch.autocast(self.device.type):
                result = self.sd_pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt="blurry, low quality, distorted, ugly",
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=1
                )
            
            # Görüntüyü al
            image = result.images[0]
            
            # Çıktı dosya yolu
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_generated_{style}_{timestamp}.png"
            output_path = f"outputs/ai_photos/{filename}"
            
            # Çıktı klasörünü oluştur
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Görüntüyü kaydet
            image.save(output_path)
            
            print(f"✅ AI fotoğraf üretildi: {output_path}")
            
            return {
                'success': True,
                'output_path': output_path,
                'prompt': prompt,
                'enhanced_prompt': enhanced_prompt,
                'style': style,
                'size': size,
                'steps': steps,
                'guidance_scale': guidance_scale,
                'seed': seed,
                'generation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'AI fotoğraf üretim hatası: {e}',
                'output_path': None
            }
    
    def generate_with_controlnet(self, prompt: str, control_image_path: str, 
                               control_type: str = 'canny', **kwargs) -> Dict:
        """ControlNet ile kontrollü fotoğraf üret"""
        if self.controlnet_pipeline_canny is None:
            return {
                'success': False,
                'error': 'ControlNet modelleri kurulu değil',
                'output_path': None
            }
        
        try:
            # Kontrol görüntüsünü yükle
            control_image = Image.open(control_image_path).convert('RGB')
            
            # Kontrol türüne göre işle
            if control_type == 'canny':
                # Canny edge detection
                control_image = self.canny_detector(control_image)
                pipeline = self.controlnet_pipeline_canny
            elif control_type == 'openpose':
                # OpenPose detection
                control_image = self.openpose_detector(control_image)
                pipeline = self.controlnet_pipeline_openpose
            else:
                return {
                    'success': False,
                    'error': f'Desteklenmeyen kontrol türü: {control_type}',
                    'output_path': None
                }
            
            print(f"🎨 ControlNet ile fotoğraf üretiliyor...")
            print(f"   Prompt: {prompt}")
            print(f"   Kontrol türü: {control_type}")
            
            # Görüntü üret
            with torch.autocast(self.device.type):
                result = pipeline(
                    prompt=prompt,
                    image=control_image,
                    negative_prompt="blurry, low quality, distorted",
                    num_inference_steps=kwargs.get('steps', 20),
                    guidance_scale=kwargs.get('guidance_scale', 7.5),
                    controlnet_conditioning_scale=kwargs.get('controlnet_scale', 1.0)
                )
            
            # Görüntüyü al
            image = result.images[0]
            
            # Çıktı dosya yolu
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"controlnet_{control_type}_{timestamp}.png"
            output_path = f"outputs/ai_photos/{filename}"
            
            # Çıktı klasörünü oluştur
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Görüntüyü kaydet
            image.save(output_path)
            
            print(f"✅ ControlNet fotoğraf üretildi: {output_path}")
            
            return {
                'success': True,
                'output_path': output_path,
                'prompt': prompt,
                'control_type': control_type,
                'control_image_path': control_image_path,
                'generation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'ControlNet fotoğraf üretim hatası: {e}',
                'output_path': None
            }
    
    def generate_variations(self, base_prompt: str, num_variations: int = 4, **kwargs) -> Dict:
        """Bir prompt'tan birden fazla varyasyon üret"""
        if self.sd_pipeline is None:
            return {
                'success': False,
                'error': 'AI modelleri kurulu değil',
                'output_paths': []
            }
        
        try:
            print(f"🎨 {num_variations} varyasyon üretiliyor...")
            
            # Çıktı klasörünü oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"outputs/ai_photos/variations_{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_paths = []
            
            for i in range(num_variations):
                # Her varyasyon için farklı seed
                seed = kwargs.get('seed', None)
                if seed is not None:
                    torch.manual_seed(seed + i)
                
                # Görüntü üret
                with torch.autocast(self.device.type):
                    result = self.sd_pipeline(
                        prompt=base_prompt,
                        negative_prompt="blurry, low quality, distorted, ugly",
                        width=kwargs.get('width', 512),
                        height=kwargs.get('height', 512),
                        num_inference_steps=kwargs.get('steps', 20),
                        guidance_scale=kwargs.get('guidance_scale', 7.5),
                        num_images_per_prompt=1
                    )
                
                # Görüntüyü kaydet
                image = result.images[0]
                output_path = output_dir / f"variation_{i+1}.png"
                image.save(output_path)
                output_paths.append(str(output_path))
                
                print(f"   Varyasyon {i+1}/{num_variations} tamamlandı")
            
            print(f"✅ {num_variations} varyasyon üretildi: {output_dir}")
            
            return {
                'success': True,
                'output_paths': output_paths,
                'output_dir': str(output_dir),
                'base_prompt': base_prompt,
                'num_variations': num_variations,
                'generation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Varyasyon üretim hatası: {e}',
                'output_paths': []
            }
    
    def upscale_image(self, image_path: str, scale_factor: int = 2) -> Dict:
        """Görüntüyü büyüt (süper çözünürlük)"""
        try:
            from PIL import Image
            import cv2
            
            # Görüntüyü yükle
            image = Image.open(image_path)
            original_size = image.size
            
            # OpenCV ile basit büyütme
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            upscaled = cv2.resize(cv_image, None, fx=scale_factor, fy=scale_factor, 
                                interpolation=cv2.INTER_CUBIC)
            
            # PIL'e geri çevir
            upscaled_image = Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
            
            # Çıktı dosya yolu
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"upscaled_{scale_factor}x_{timestamp}.png"
            output_path = f"outputs/ai_photos/{filename}"
            
            # Çıktı klasörünü oluştur
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Görüntüyü kaydet
            upscaled_image.save(output_path)
            
            print(f"✅ Görüntü büyütüldü: {original_size} -> {upscaled_image.size}")
            
            return {
                'success': True,
                'output_path': output_path,
                'original_size': original_size,
                'upscaled_size': upscaled_image.size,
                'scale_factor': scale_factor,
                'generation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Görüntü büyütme hatası: {e}',
                'output_path': None
            }
    
    def get_available_styles(self) -> List[Dict]:
        """Mevcut stilleri al"""
        styles = [
            {
                'id': 'realistic',
                'name': 'Gerçekçi',
                'description': 'Fotoğraf gerçekçi stil',
                'keywords': 'photorealistic, high quality, detailed'
            },
            {
                'id': 'anime',
                'name': 'Anime',
                'description': 'Anime/manga stil',
                'keywords': 'anime style, manga style, japanese animation'
            },
            {
                'id': 'oil_painting',
                'name': 'Yağlı Boya',
                'description': 'Yağlı boya tablo stili',
                'keywords': 'oil painting, classical art, renaissance style'
            },
            {
                'id': 'watercolor',
                'name': 'Sulu Boya',
                'description': 'Sulu boya resim stili',
                'keywords': 'watercolor painting, soft colors, artistic'
            },
            {
                'id': 'sketch',
                'name': 'Çizim',
                'description': 'Karakalem çizim stili',
                'keywords': 'pencil sketch, black and white, line art'
            },
            {
                'id': 'cyberpunk',
                'name': 'Cyberpunk',
                'description': 'Futuristik cyberpunk stili',
                'keywords': 'cyberpunk, futuristic, neon lights, sci-fi'
            },
            {
                'id': 'fantasy',
                'name': 'Fantastik',
                'description': 'Fantastik sanat stili',
                'keywords': 'fantasy art, magical, mystical, ethereal'
            },
            {
                'id': 'portrait',
                'name': 'Portre',
                'description': 'Profesyonel portre stili',
                'keywords': 'professional portrait, studio lighting, high resolution'
            }
        ]
        
        return styles

def main():
    """Test fonksiyonu"""
    generator = AIPhotoGenerator()
    
    if generator.sd_pipeline is None:
        print("❌ AI modelleri kurulu değil!")
        print("💡 Kurulum için: pip install diffusers transformers controlnet-aux")
        return
    
    # Test fotoğraf üretimi
    result = generator.generate_image(
        prompt="a beautiful landscape with mountains and lake",
        style="realistic",
        size="512x512",
        steps=20
    )
    
    if result['success']:
        print(f"✅ Test fotoğraf üretildi: {result['output_path']}")
    else:
        print(f"❌ Test hatası: {result['error']}")

if __name__ == "__main__":
    main()
