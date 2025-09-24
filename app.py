#!/usr/bin/env python3
"""
Face/Video AI Studio - Ana Uygulama
FastAPI tabanlı web arayüzü ve API
"""

import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.requests import Request
from pydantic import BaseModel
import redis
from rq import Queue
import uvicorn
import uuid

# Proje kök dizinini Python path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.database import get_db_manager
from core.config import get_config_manager, JobConfig
from workers.tasks import process_job_task
from core.websocket_manager import get_connection_manager, get_websocket_handler, get_job_tracker

# FastAPI uygulaması
app = FastAPI(
    title="Face/Video AI Studio",
    description="Yapay zeka destekli fotoğraf ve video işleme platformu",
    version="1.0.0"
)

# Statik dosyalar ve template'ler
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

# Redis ve RQ kuyruğu
redis_conn = redis.Redis(host='localhost', port=6379, db=0)
job_queue = Queue('default', connection=redis_conn)

# Global yöneticiler
db_manager = get_db_manager()
config_manager = get_config_manager()
connection_manager = get_connection_manager()
websocket_handler = get_websocket_handler()
job_tracker = get_job_tracker()

# Pydantic modelleri
class JobCreate(BaseModel):
    job_type: str
    inputs: Dict[str, str]
    params: Dict[str, Any] = {}
    preset: Optional[str] = None
    consent_tag: str = "unknown"

class JobResponse(BaseModel):
    id: int
    status: str
    created_at: datetime
    progress: float = 0.0
    current_step: Optional[str] = None

# Ana sayfa
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard ana sayfa"""
    recent_jobs = db_manager.list_jobs(limit=10)
    presets = config_manager.list_presets()
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "recent_jobs": recent_jobs,
        "presets": presets
    })

@app.get("/new")
async def new_project(request: Request, type: Optional[str] = None, preset: Optional[str] = None):
    """Yeni proje oluşturma sayfası"""
    try:
        presets = config_manager.list_presets()
        return templates.TemplateResponse("enhanced_smart_wizard.html", {
            "request": request,
            "presets": presets,
            "default_type": type,
            "default_preset": preset
        })
    except Exception as e:
        return templates.TemplateResponse("enhanced_smart_wizard.html", {
            "request": request,
            "presets": [],
            "default_type": type,
            "default_preset": preset
        })

@app.get("/gallery")
async def gallery(request: Request):
    """Galeri sayfası"""
    try:
        jobs = db_manager.list_jobs(limit=50)
        return templates.TemplateResponse("enhanced_gallery.html", {
            "request": request,
            "jobs": jobs
        })
    except Exception as e:
        return templates.TemplateResponse("enhanced_gallery.html", {
            "request": request,
            "jobs": []
        })

@app.get("/job/{job_id}")
async def job_detail(request: Request, job_id: int):
    """İş detay sayfası"""
    try:
        job = db_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="İş bulunamadı")
        
        return templates.TemplateResponse("job_detail.html", {
            "request": request,
            "job": job
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API Endpoints
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Dosya yükleme"""
    try:
        system_config = config_manager.get_system_config()
        max_size = system_config.get('max_file_size_mb', 100) * 1024 * 1024
        
        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(status_code=413, detail="Dosya çok büyük")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        upload_path = Path("storage/uploads") / filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(upload_path, "wb") as f:
            f.write(content)
        
        return {
            "success": True,
            "filename": filename,
            "filepath": str(upload_path),
            "size": len(content)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jobs", response_model=JobResponse)
async def create_job(job_data: JobCreate):
    """Yeni iş oluştur"""
    try:
        job_config = config_manager.create_job_config(
            job_type=job_data.job_type,
            inputs=job_data.inputs,
            params=job_data.params,
            preset=job_data.preset
        )
        
        errors = config_manager.validate_config(job_config)
        if errors:
            raise HTTPException(status_code=400, detail=f"Konfigürasyon hatası: {'; '.join(errors)}")
        
        yaml_config = yaml.dump(job_config.dict(), default_flow_style=False, allow_unicode=True)
        
        job = db_manager.create_job(
            job_type=job_data.job_type,
            yaml_config=yaml_config,
            consent_tag=job_data.consent_tag
        )
        
        job_queue.enqueue(process_job_task, job.id)
        
        # WebSocket bildirimi
        await connection_manager.broadcast({
            "type": "job_created",
            "job_id": job.id,
            "job_type": job.job_type,
            "timestamp": datetime.now().isoformat()
        })
        
        # İş takibine ekle
        await job_tracker.add_job_to_tracking(job.id)
        
        return JobResponse(
            id=job.id,
            status=job.status,
            created_at=job.created_at,
            progress=job.progress,
            current_step=job.current_step
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs")
async def get_jobs(limit: int = 10, offset: int = 0):
    """İş listesini al"""
    try:
        jobs = db_manager.list_jobs(limit=limit, offset=offset)
        return {
            "jobs": [
                {
                    "id": job.id,
                    "job_type": job.job_type,
                    "status": job.status,
                    "created_at": job.created_at,
                    "progress": job.progress,
                    "current_step": job.current_step
                }
                for job in jobs
            ],
            "total": len(jobs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: int):
    """İş bilgisini al"""
    job = db_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="İş bulunamadı")
    
    return {
        "id": job.id,
        "job_type": job.job_type,
        "status": job.status,
        "progress": job.progress,
        "current_step": job.current_step,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "output_path": job.output_path,
        "error_message": job.error_message,
        "metrics": job.metrics_json
    }

@app.get("/api/presets")
async def get_presets():
    """Preset'leri al"""
    presets = config_manager.list_presets()
    preset_data = []
    
    for preset_name in presets:
        preset_config = config_manager.get_preset(preset_name)
        if preset_config:
            preset_data.append({
                "name": preset_name,
                "description": preset_config.get("description", ""),
                "job_type": preset_config.get("job_type", ""),
                "params": preset_config.get("params", {})
            })
    
    return {"presets": preset_data}

# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket bağlantı endpoint'i"""
    user_id = None  # Gerçek uygulamada authentication'dan alınacak
    
    try:
        await connection_manager.connect(websocket, client_id, user_id)
        
        while True:
            # Mesaj bekle
            data = await websocket.receive_text()
            await websocket_handler.handle_message(websocket, data, client_id, user_id)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, client_id, user_id)
    except Exception as e:
        print(f"WebSocket hatası: {e}")
        connection_manager.disconnect(websocket, client_id, user_id)

if __name__ == "__main__":
    print("🚀 Face/Video AI Studio başlatılıyor...")
    print("📱 Web arayüzü: http://localhost:8000")
    print("🔧 API: http://localhost:8000/docs")
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)