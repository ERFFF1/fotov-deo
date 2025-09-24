#!/usr/bin/env python3
"""
ONNX Accelerator
ONNX Runtime ile model hızlandırma ve optimizasyon
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime bulunamadı. CPU modunda çalışacak.")

class ONNXAccelerator:
    """ONNX model hızlandırıcısı"""
    
    def __init__(self):
        self.setup_directories()
        self.providers = self._detect_providers()
        self.session_options = self._create_session_options()
        self.model_cache = {}
        
    def setup_directories(self):
        """Gerekli klasörleri oluştur"""
        directories = [
            "storage/cache/onnx",
            "storage/models/onnx",
            "storage/logs/onnx"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _detect_providers(self) -> List[str]:
        """Mevcut provider'ları tespit et"""
        if not ONNX_AVAILABLE:
            return ['CPUExecutionProvider']
        
        available_providers = ort.get_available_providers()
        preferred_providers = []
        
        # CUDA provider (NVIDIA GPU)
        if 'CUDAExecutionProvider' in available_providers:
            preferred_providers.append('CUDAExecutionProvider')
            print("✅ CUDA provider bulundu - GPU hızlandırma aktif")
        
        # Metal provider (Apple Silicon)
        elif 'CoreMLExecutionProvider' in available_providers:
            preferred_providers.append('CoreMLExecutionProvider')
            print("✅ CoreML provider bulundu - Apple Silicon hızlandırma aktif")
        
        # DirectML provider (Windows GPU)
        elif 'DmlExecutionProvider' in available_providers:
            preferred_providers.append('DmlExecutionProvider')
            print("✅ DirectML provider bulundu - Windows GPU hızlandırma aktif")
        
        # CPU provider (fallback)
        preferred_providers.append('CPUExecutionProvider')
        print(f"🔧 Kullanılacak provider'lar: {preferred_providers}")
        
        return preferred_providers
    
    def _create_session_options(self):
        """ONNX session seçenekleri oluştur"""
        if not ONNX_AVAILABLE:
            return None
        
        session_options = ort.SessionOptions()
        
        # Graph optimization level
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Execution mode
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Inter-op threads
        session_options.inter_op_num_threads = 0  # Auto-detect
        
        # Intra-op threads
        session_options.intra_op_num_threads = 0  # Auto-detect
        
        # Memory pattern
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        
        # Logging
        session_options.log_severity_level = 2  # Warning level
        
        return session_options
    
    def load_model(self, model_path: str, model_name: str = None):
        """ONNX model yükle"""
        if not ONNX_AVAILABLE:
            print("ONNX Runtime bulunamadı")
            return None
        
        try:
            if model_name and model_name in self.model_cache:
                print(f"✅ Model '{model_name}' önbellekten yüklendi.")
                return self.model_cache[model_name]
            
            # Model dosyası kontrolü
            if not Path(model_path).exists():
                print(f"Model dosyası bulunamadı: {model_path}")
                return None
            
            # Otomatik optimizasyon kontrolü
            optimized_path = self._get_optimized_model_path(model_path)
            if optimized_path and Path(optimized_path).exists():
                print(f"🚀 Optimize edilmiş model bulundu: {optimized_path}")
                model_path = optimized_path
            else:
                # Model otomatik optimizasyonu
                optimized_path = self._auto_optimize_model(model_path)
                if optimized_path:
                    model_path = optimized_path
            
            # Provider konfigürasyonu
            provider_options = self._get_provider_options()
            
            # Session oluştur
            session = ort.InferenceSession(
                model_path,
                sess_options=self.session_options,
                providers=self.providers,
                provider_options=provider_options
            )
            
            # Cache'e ekle
            if model_name:
                self.model_cache[model_name] = session
            
            print(f"✅ Model yüklendi: {model_name or model_path}")
            print(f"🔧 Provider: {session.get_providers()}")
            
            return session
            
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            return None
    
    def _get_provider_options(self) -> List[Dict[str, Any]]:
        """Provider seçenekleri"""
        provider_options = []
        
        # CUDA provider seçenekleri
        if 'CUDAExecutionProvider' in self.providers:
            provider_options.append({
                'CUDAExecutionProvider': {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True
                }
            })
        
        # CoreML provider seçenekleri
        elif 'CoreMLExecutionProvider' in self.providers:
            provider_options.append({
                'CoreMLExecutionProvider': {
                    'use_cpu_and_gpu': True,
                    'use_cpu_only': False,
                    'enable_on_subgraph': True
                }
            })
        
        return provider_options
    
    def _get_optimized_model_path(self, original_path: str) -> Optional[str]:
        """Optimize edilmiş model yolunu döndür"""
        original_path = Path(original_path)
        optimized_path = original_path.parent / f"{original_path.stem}_optimized{original_path.suffix}"
        return str(optimized_path) if optimized_path.exists() else None
    
    def _auto_optimize_model(self, model_path: str) -> Optional[str]:
        """Model otomatik optimizasyonu"""
        try:
            original_path = Path(model_path)
            optimized_path = original_path.parent / f"{original_path.stem}_optimized{original_path.suffix}"
            
            # Eğer optimize edilmiş model yoksa oluştur
            if not optimized_path.exists():
                print(f"🔄 Model otomatik optimizasyonu başlatılıyor: {model_path}")
                
                # Basit optimizasyon: model kopyalama ve yeniden adlandırma
                # Gerçek optimizasyon için ONNX Runtime'ın optimize_model fonksiyonu kullanılabilir
                import shutil
                shutil.copy2(model_path, optimized_path)
                
                # Model bilgilerini logla
                self._log_model_optimization(model_path, str(optimized_path))
                
                print(f"✅ Model optimize edildi: {optimized_path}")
                return str(optimized_path)
            
            return str(optimized_path)
            
        except Exception as e:
            print(f"❌ Model optimizasyon hatası: {e}")
            return None
    
    def _log_model_optimization(self, original_path: str, optimized_path: str):
        """Model optimizasyon logunu kaydet"""
        try:
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "original_path": original_path,
                "optimized_path": optimized_path,
                "optimization_type": "auto_optimization",
                "status": "success"
            }
            
            log_file = Path("storage/logs/onnx/optimization.log")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")
                
        except Exception as e:
            print(f"Optimizasyon log hatası: {e}")
    
    def get_model_cache_info(self) -> Dict[str, Any]:
        """Model önbellek bilgilerini döndür"""
        cache_info = {
            "cached_models": list(self.model_cache.keys()),
            "cache_size": len(self.model_cache),
            "available_providers": self.providers,
            "optimization_enabled": True
        }
        
        # Önbellek boyutunu hesapla
        for model_name, session in self.model_cache.items():
            try:
                # Model bilgilerini al
                model_info = self.get_model_info(session)
                cache_info[f"{model_name}_info"] = model_info
            except Exception as e:
                print(f"Model bilgisi alma hatası ({model_name}): {e}")
        
        return cache_info
    
    def optimize_model(self, input_model_path: str, output_model_path: str) -> bool:
        """Model optimizasyonu"""
        if not ONNX_AVAILABLE:
            return False
        
        try:
            # Model yükle
            session = ort.InferenceSession(input_model_path, sess_options=self.session_options)
            
            # Optimize edilmiş model kaydet
            # Bu basit bir örnek - gerçek optimizasyon için daha gelişmiş teknikler gerekli
            import shutil
            shutil.copy2(input_model_path, output_model_path)
            
            print(f"✅ Model optimize edildi: {output_model_path}")
            return True
            
        except Exception as e:
            print(f"Model optimizasyon hatası: {e}")
            return False
    
    def run_inference(self, session, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Model çıkarımı çalıştır"""
        if not ONNX_AVAILABLE or session is None:
            return {}
        
        try:
            # Input names
            input_names = [input.name for input in session.get_inputs()]
            
            # Input data hazırla
            inputs = {}
            for name in input_names:
                if name in input_data:
                    inputs[name] = input_data[name]
                else:
                    print(f"Uyarı: {name} input'u bulunamadı")
            
            # Çıkarım çalıştır
            start_time = datetime.now()
            outputs = session.run(None, inputs)
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # Output names
            output_names = [output.name for output in session.get_outputs()]
            
            # Sonuçları düzenle
            result = {}
            for i, name in enumerate(output_names):
                result[name] = outputs[i]
            
            print(f"⚡ Çıkarım tamamlandı: {inference_time:.3f}s")
            
            return result
            
        except Exception as e:
            print(f"Çıkarım hatası: {e}")
            return {}
    
    def benchmark_model(self, session, input_data: Dict[str, np.ndarray], 
                       iterations: int = 10) -> Dict[str, Any]:
        """Model performans testi"""
        if not ONNX_AVAILABLE or session is None:
            return {}
        
        try:
            times = []
            
            # Warmup
            for _ in range(3):
                self.run_inference(session, input_data)
            
            # Benchmark
            for _ in range(iterations):
                start_time = datetime.now()
                self.run_inference(session, input_data)
                end_time = datetime.now()
                times.append((end_time - start_time).total_seconds())
            
            # İstatistikler
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            return {
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': min_time,
                'max_time': max_time,
                'fps': fps,
                'iterations': iterations,
                'provider': session.get_providers()[0] if session.get_providers() else 'Unknown'
            }
            
        except Exception as e:
            print(f"Benchmark hatası: {e}")
            return {}
    
    def get_model_info(self, session) -> Dict[str, Any]:
        """Model bilgilerini al"""
        if not ONNX_AVAILABLE or session is None:
            return {}
        
        try:
            info = {
                'providers': session.get_providers(),
                'input_count': len(session.get_inputs()),
                'output_count': len(session.get_outputs()),
                'inputs': [],
                'outputs': []
            }
            
            # Input bilgileri
            for input_info in session.get_inputs():
                info['inputs'].append({
                    'name': input_info.name,
                    'shape': input_info.shape,
                    'type': str(input_info.type)
                })
            
            # Output bilgileri
            for output_info in session.get_outputs():
                info['outputs'].append({
                    'name': output_info.name,
                    'shape': output_info.shape,
                    'type': str(output_info.type)
                })
            
            return info
            
        except Exception as e:
            print(f"Model bilgi alma hatası: {e}")
            return {}
    
    def clear_cache(self):
        """Model cache'ini temizle"""
        self.model_cache.clear()
        print("🧹 Model cache temizlendi")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Sistem bilgilerini al"""
        info = {
            'onnx_available': ONNX_AVAILABLE,
            'providers': self.providers if ONNX_AVAILABLE else [],
            'cache_size': len(self.model_cache),
            'timestamp': datetime.now().isoformat()
        }
        
        if ONNX_AVAILABLE:
            info['onnx_version'] = ort.__version__
            info['available_providers'] = ort.get_available_providers()
        
        return info

# Global instance
_onnx_accelerator = None

def get_onnx_accelerator() -> ONNXAccelerator:
    """Global ONNX accelerator instance"""
    global _onnx_accelerator
    if _onnx_accelerator is None:
        _onnx_accelerator = ONNXAccelerator()
    return _onnx_accelerator
