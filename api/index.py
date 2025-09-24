#!/usr/bin/env python3
"""
Face/Video AI Studio - Vercel API Entry Point
Hafif FastAPI uygulaması - ML importları yok
"""

import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
import redis
from rq import Queue
import json

# FastAPI uygulaması
app = FastAPI(
    title="Face/Video AI Studio",
    description="AI-powered photo and video processing platform",
    version="1.0.0"
)

# Template ve static dosyalar
templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Redis bağlantısı (Upstash)
def get_redis_connection():
    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        return redis.from_url(redis_url)
    except Exception as e:
        print(f"Redis bağlantı hatası: {e}")
        return None

def get_job_queue():
    try:
        redis_conn = get_redis_connection()
        if redis_conn:
            return Queue('default', connection=redis_conn)
        return None
    except Exception as e:
        print(f"Queue oluşturma hatası: {e}")
        return None

# Ana sayfa
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Ana dashboard sayfası"""
    try:
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "recent_jobs": [],
            "presets": [
                {"name": "face_swap_basic", "description": "Temel yüz değiştirme"},
                {"name": "talking_wav2lip", "description": "Wav2Lip konuşan kafa"},
                {"name": "color_grade_rec709", "description": "Rec709 renk düzeltme"}
            ]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# API endpoints
@app.get("/api/health")
async def health_check():
    """Sistem durumu kontrolü"""
    return {
        "status": "healthy",
        "service": "face-video-ai-studio",
        "version": "1.0.0",
        "redis_connected": get_redis_connection() is not None
    }

@app.get("/api/presets")
async def get_presets():
    """Mevcut preset'leri listele"""
    presets = [
        {
            "name": "face_swap_basic",
            "description": "Temel yüz değiştirme - yüksek kalite",
            "job_type": "face_swap",
            "params": {
                "detector": "insightface",
                "swapper": "inswapper_128.onnx",
                "quality": "high"
            }
        },
        {
            "name": "talking_wav2lip",
            "description": "Wav2Lip ile konuşan kafa - profesyonel kalite",
            "job_type": "talking_head",
            "params": {
                "method": "wav2lip",
                "enhancer": "gfpgan",
                "quality": "high"
            }
        },
        {
            "name": "color_grade_rec709",
            "description": "Rec709 renk düzeltme - sinema kalitesi",
            "job_type": "color_grade",
            "params": {
                "lut": "rec709.cube",
                "saturation": 1.05,
                "contrast": 1.02
            }
        }
    ]
    return {"presets": presets}

@app.get("/api/jobs")
async def get_jobs(limit: int = 10, offset: int = 0):
    """İş listesini al"""
    # Gerçek uygulamada Redis'ten alınacak
    return {
        "jobs": [],
        "total": 0,
        "message": "Worker sistemi aktif değil"
    }

@app.post("/api/jobs")
async def create_job(job_data: dict):
    """Yeni iş oluştur ve kuyruğa ekle"""
    try:
        queue = get_job_queue()
        if not queue:
            return JSONResponse({
                "error": "Redis bağlantısı yok",
                "queued": False
            }, status_code=500)
        
        # İşi kuyruğa ekle
        job = queue.enqueue(
            "workers.tasks.process_job_task",
            job_data,
            timeout=300  # 5 dakika timeout
        )
        
        return {
            "queued": True,
            "job_id": job.id,
            "status": "pending",
            "message": "İş worker'a gönderildi"
        }
        
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "queued": False
        }, status_code=500)

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """İş durumunu kontrol et"""
    try:
        queue = get_job_queue()
        if not queue:
            return JSONResponse({
                "error": "Redis bağlantısı yok"
            }, status_code=500)
        
        job = queue.fetch_job(job_id)
        if not job:
            return JSONResponse({
                "error": "İş bulunamadı"
            }, status_code=404)
        
        return {
            "id": job_id,
            "status": job.get_status(),
            "result": job.result,
            "created_at": job.created_at.isoformat() if job.created_at else None
        }
        
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)

# Web sayfaları
@app.get("/new")
async def new_project(request: Request):
    """Yeni proje sayfası"""
    try:
        return templates.TemplateResponse("enhanced_smart_wizard.html", {
            "request": request,
            "presets": [],
            "default_type": None,
            "default_preset": None
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/gallery")
async def gallery(request: Request):
    """Galeri sayfası"""
    try:
        return templates.TemplateResponse("enhanced_gallery.html", {
            "request": request,
            "jobs": []
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Vercel için gerekli
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
