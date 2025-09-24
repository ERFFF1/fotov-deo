#!/usr/bin/env python3
"""
Device Manager
GPU yönetimi ve model önbellekleme sistemi
"""

import os
import torch
import psutil
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import threading
import time

class DeviceManager:
    """GPU ve CPU kaynak yöneticisi"""
    
    def __init__(self):
        self.devices = self._detect_devices()
        self.model_cache = {}
        self.cache_lock = threading.Lock()
        
    def _detect_devices(self) -> Dict[str, Any]:
        """Mevcut cihazları tespit et"""
        devices = {
            'cpu': {
                'available': True,
                'cores': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'current_load': 0.0
            }
        }
        
        # GPU tespiti
        if torch.cuda.is_available():
            devices['gpu'] = {
                'available': True,
                'count': torch.cuda.device_count(),
                'devices': []
            }
            
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'memory_allocated': 0,
                    'memory_free': 0,
                    'utilization': 0.0,
                    'temperature': 0.0,
                    'in_use': False
                }
                devices['gpu']['devices'].append(gpu_info)
        else:
            devices['gpu'] = {'available': False}
        
        return devices
    
    def get_optimal_device(self, job_type: str, memory_requirement_gb: float = 2.0) -> str:
        """İş türüne göre en uygun cihazı seç"""
        if not self.devices['gpu']['available']:
            return 'cpu'
        
        # GPU gerektiren işler
        gpu_intensive_jobs = [
            'face_enhance', 'face_swap', 'talking_head', 
            'ai_photo_generation', 'ai_video_generation'
        ]
        
        if job_type not in gpu_intensive_jobs:
            return 'cpu'
        
        # En uygun GPU'yu bul
        best_gpu = None
        best_score = -1
        
        for gpu in self.devices['gpu']['devices']:
            if gpu['in_use']:
                continue
                
            free_memory_gb = gpu['memory_free'] / (1024**3)
            if free_memory_gb < memory_requirement_gb:
                continue
            
            # Skor hesapla (bellek + performans)
            score = free_memory_gb * (1 - gpu['utilization'])
            
            if score > best_score:
                best_score = score
                best_gpu = gpu
        
        if best_gpu:
            return f"cuda:{best_gpu['id']}"
        else:
            return 'cpu'
    
    def allocate_device(self, device_id: str) -> bool:
        """Cihazı ayır"""
        if device_id.startswith('cuda:'):
            gpu_id = int(device_id.split(':')[1])
            if gpu_id < len(self.devices['gpu']['devices']):
                self.devices['gpu']['devices'][gpu_id]['in_use'] = True
                return True
        return False
    
    def release_device(self, device_id: str):
        """Cihazı serbest bırak"""
        if device_id.startswith('cuda:'):
            gpu_id = int(device_id.split(':')[1])
            if gpu_id < len(self.devices['gpu']['devices']):
                self.devices['gpu']['devices'][gpu_id]['in_use'] = False
    
    def update_device_stats(self):
        """Cihaz istatistiklerini güncelle"""
        # CPU istatistikleri
        self.devices['cpu']['current_load'] = psutil.cpu_percent()
        
        # GPU istatistikleri
        if self.devices['gpu']['available']:
            for gpu in self.devices['gpu']['devices']:
                try:
                    torch.cuda.set_device(gpu['id'])
                    gpu['memory_allocated'] = torch.cuda.memory_allocated(gpu['id'])
                    gpu['memory_free'] = torch.cuda.memory_reserved(gpu['id']) - gpu['memory_allocated']
                    
                    # GPU kullanım oranı (basit hesaplama)
                    total_memory = gpu['memory_total']
                    used_memory = gpu['memory_allocated']
                    gpu['utilization'] = used_memory / total_memory if total_memory > 0 else 0
                    
                except Exception as e:
                    print(f"GPU {gpu['id']} istatistik güncelleme hatası: {e}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Cihaz bilgilerini al"""
        self.update_device_stats()
        return self.devices

class ModelCache:
    """Model önbellekleme sistemi"""
    
    def __init__(self, cache_dir: str = "storage/cache/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index = {}
        self.load_cache_index()
        
    def load_cache_index(self):
        """Önbellek indeksini yükle"""
        index_file = self.cache_dir / "cache_index.pkl"
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    self.cache_index = pickle.load(f)
            except Exception as e:
                print(f"Önbellek indeksi yükleme hatası: {e}")
                self.cache_index = {}
    
    def save_cache_index(self):
        """Önbellek indeksini kaydet"""
        index_file = self.cache_dir / "cache_index.pkl"
        try:
            with open(index_file, 'wb') as f:
                pickle.dump(self.cache_index, f)
        except Exception as e:
            print(f"Önbellek indeksi kaydetme hatası: {e}")
    
    def get_cache_key(self, model_name: str, model_version: str, device: str) -> str:
        """Önbellek anahtarı oluştur"""
        key_string = f"{model_name}_{model_version}_{device}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def is_cached(self, model_name: str, model_version: str, device: str) -> bool:
        """Model önbellekte mi kontrol et"""
        cache_key = self.get_cache_key(model_name, model_version, device)
        return cache_key in self.cache_index
    
    def get_cached_model(self, model_name: str, model_version: str, device: str) -> Optional[Any]:
        """Önbellekten model al"""
        cache_key = self.get_cache_key(model_name, model_version, device)
        
        if cache_key not in self.cache_index:
            return None
        
        cache_info = self.cache_index[cache_key]
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            # Dosya yoksa indeksten kaldır
            del self.cache_index[cache_key]
            self.save_cache_index()
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                model = pickle.load(f)
            
            # Son erişim zamanını güncelle
            cache_info['last_accessed'] = datetime.now()
            self.save_cache_index()
            
            return model
        except Exception as e:
            print(f"Önbellekten model yükleme hatası: {e}")
            return None
    
    def cache_model(self, model_name: str, model_version: str, device: str, model: Any) -> bool:
        """Modeli önbelleğe kaydet"""
        cache_key = self.get_cache_key(model_name, model_version, device)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(model, f)
            
            # İndekse ekle
            self.cache_index[cache_key] = {
                'model_name': model_name,
                'model_version': model_version,
                'device': device,
                'cached_at': datetime.now(),
                'last_accessed': datetime.now(),
                'file_size': cache_file.stat().st_size
            }
            
            self.save_cache_index()
            return True
            
        except Exception as e:
            print(f"Model önbellekleme hatası: {e}")
            return False
    
    def cleanup_old_cache(self, max_age_days: int = 7, max_size_gb: float = 10.0):
        """Eski önbellek dosyalarını temizle"""
        current_time = datetime.now()
        max_age = timedelta(days=max_age_days)
        max_size_bytes = max_size_gb * 1024**3
        
        # Eski dosyaları bul
        to_remove = []
        total_size = 0
        
        for cache_key, cache_info in self.cache_index.items():
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if not cache_file.exists():
                to_remove.append(cache_key)
                continue
            
            file_age = current_time - cache_info['last_accessed']
            if file_age > max_age:
                to_remove.append(cache_key)
                continue
            
            total_size += cache_info['file_size']
        
        # Boyut limitini aşan dosyaları bul
        if total_size > max_size_bytes:
            # En az kullanılan dosyaları sırala
            sorted_items = sorted(
                self.cache_index.items(),
                key=lambda x: x[1]['last_accessed']
            )
            
            for cache_key, cache_info in sorted_items:
                if total_size <= max_size_bytes:
                    break
                
                to_remove.append(cache_key)
                total_size -= cache_info['file_size']
        
        # Dosyaları kaldır
        for cache_key in to_remove:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            
            if cache_key in self.cache_index:
                del self.cache_index[cache_key]
        
        if to_remove:
            self.save_cache_index()
            print(f"Önbellek temizlendi: {len(to_remove)} dosya kaldırıldı")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Önbellek istatistiklerini al"""
        total_files = len(self.cache_index)
        total_size = sum(info['file_size'] for info in self.cache_index.values())
        
        return {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_gb': total_size / (1024**3),
            'models': list(set(info['model_name'] for info in self.cache_index.values()))
        }

class PerformanceMonitor:
    """Performans izleme sistemi"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def start_timer(self, operation: str) -> str:
        """Zamanlayıcı başlat"""
        timer_id = f"{operation}_{int(time.time() * 1000)}"
        self.metrics[timer_id] = {
            'operation': operation,
            'start_time': time.time(),
            'end_time': None,
            'duration': None,
            'device': None,
            'memory_used': None
        }
        return timer_id
    
    def end_timer(self, timer_id: str, device: str = None, memory_used: float = None):
        """Zamanlayıcı bitir"""
        if timer_id in self.metrics:
            end_time = time.time()
            self.metrics[timer_id]['end_time'] = end_time
            self.metrics[timer_id]['duration'] = end_time - self.metrics[timer_id]['start_time']
            self.metrics[timer_id]['device'] = device
            self.metrics[timer_id]['memory_used'] = memory_used
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """İşlem istatistiklerini al"""
        operation_metrics = [m for m in self.metrics.values() if m['operation'] == operation]
        
        if not operation_metrics:
            return {}
        
        durations = [m['duration'] for m in operation_metrics if m['duration'] is not None]
        
        return {
            'count': len(operation_metrics),
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'total_duration': sum(durations)
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Tüm istatistikleri al"""
        operations = set(m['operation'] for m in self.metrics.values())
        
        return {
            'uptime': time.time() - self.start_time,
            'total_operations': len(self.metrics),
            'operations': {op: self.get_operation_stats(op) for op in operations}
        }

# Global instances
device_manager = DeviceManager()
model_cache = ModelCache()
performance_monitor = PerformanceMonitor()

def get_device_manager() -> DeviceManager:
    """Device manager'ı al"""
    return device_manager

def get_model_cache() -> ModelCache:
    """Model cache'i al"""
    return model_cache

def get_performance_monitor() -> PerformanceMonitor:
    """Performance monitor'ı al"""
    return performance_monitor
