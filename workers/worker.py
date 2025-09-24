#!/usr/bin/env python3
"""
Face/Video AI Studio - Worker Process
Ağır ML işlemlerini yapan worker süreci
"""

import os
import sys
import redis
from rq import Worker, Connection
from rq.worker import WorkerStatus

# Proje kök dizinini Python path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def start_worker():
    """Worker sürecini başlat"""
    try:
        # Redis bağlantısı
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        redis_conn = redis.from_url(redis_url)
        
        # Worker oluştur
        with Connection(redis_conn):
            worker = Worker(['default'])
            print("🚀 Face/Video AI Studio Worker başlatılıyor...")
            print(f"📡 Redis URL: {redis_url}")
            print("⏳ İş bekleniyor...")
            worker.work()
            
    except Exception as e:
        print(f"❌ Worker başlatma hatası: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_worker()