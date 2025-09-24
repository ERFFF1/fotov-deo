#!/usr/bin/env python3
"""
RQ Worker
Redis tabanlı iş kuyruğu worker'ı
"""

import os
import sys
from pathlib import Path

# Proje kök dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rq import Worker, Connection
import redis

def main():
    """Worker ana fonksiyonu"""
    # Redis bağlantısı
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    redis_conn = redis.from_url(redis_url)
    
    # Worker'ı başlat
    with Connection(redis_conn):
        worker = Worker(['default'])
        print("🚀 RQ Worker başlatılıyor...")
        print(f"📡 Redis: {redis_url}")
        print("⏳ İş bekleniyor...")
        worker.work()

if __name__ == '__main__':
    main()
