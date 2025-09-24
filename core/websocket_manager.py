#!/usr/bin/env python3
"""
WebSocket Manager
Gerçek zamanlı güncellemeler için WebSocket yönetimi
"""

import asyncio
import json
from typing import Dict, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import logging

class ConnectionManager:
    """WebSocket bağlantı yöneticisi"""
    
    def __init__(self):
        # Aktif bağlantılar
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Kullanıcı bağlantıları
        self.user_connections: Dict[str, Set[WebSocket]] = {}
        # İş takipçileri
        self.job_watchers: Dict[int, Set[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str, user_id: str = None):
        """Bağlantı kur"""
        await websocket.accept()
        
        # Genel bağlantı listesine ekle
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()
        self.active_connections[client_id].add(websocket)
        
        # Kullanıcı bağlantılarına ekle
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(websocket)
        
        # Bağlantı bilgilerini gönder
        await self.send_personal_message({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat()
        }, websocket)
        
        logging.info(f"WebSocket bağlantısı kuruldu: {client_id}")
    
    def disconnect(self, websocket: WebSocket, client_id: str, user_id: str = None):
        """Bağlantıyı kes"""
        # Genel bağlantı listesinden çıkar
        if client_id in self.active_connections:
            self.active_connections[client_id].discard(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
        
        # Kullanıcı bağlantılarından çıkar
        if user_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # İş takipçilerinden çıkar
        for job_id, watchers in self.job_watchers.items():
            watchers.discard(websocket)
            if not watchers:
                del self.job_watchers[job_id]
        
        logging.info(f"WebSocket bağlantısı kesildi: {client_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Kişisel mesaj gönder"""
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logging.error(f"Kişisel mesaj gönderme hatası: {e}")
    
    async def send_to_client(self, message: Dict[str, Any], client_id: str):
        """Belirli istemciye mesaj gönder"""
        if client_id in self.active_connections:
            disconnected = set()
            for websocket in self.active_connections[client_id]:
                try:
                    await websocket.send_text(json.dumps(message, ensure_ascii=False))
                except Exception as e:
                    logging.error(f"İstemci mesajı gönderme hatası: {e}")
                    disconnected.add(websocket)
            
            # Bağlantısı kesilen websocket'leri temizle
            for websocket in disconnected:
                self.active_connections[client_id].discard(websocket)
    
    async def send_to_user(self, message: Dict[str, Any], user_id: str):
        """Belirli kullanıcıya mesaj gönder"""
        if user_id in self.user_connections:
            disconnected = set()
            for websocket in self.user_connections[user_id]:
                try:
                    await websocket.send_text(json.dumps(message, ensure_ascii=False))
                except Exception as e:
                    logging.error(f"Kullanıcı mesajı gönderme hatası: {e}")
                    disconnected.add(websocket)
            
            # Bağlantısı kesilen websocket'leri temizle
            for websocket in disconnected:
                self.user_connections[user_id].discard(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Tüm bağlantılara yayın yap"""
        disconnected = set()
        
        for client_id, connections in self.active_connections.items():
            for websocket in connections:
                try:
                    await websocket.send_text(json.dumps(message, ensure_ascii=False))
                except Exception as e:
                    logging.error(f"Yayın mesajı gönderme hatası: {e}")
                    disconnected.add((client_id, websocket))
        
        # Bağlantısı kesilen websocket'leri temizle
        for client_id, websocket in disconnected:
            self.active_connections[client_id].discard(websocket)
    
    def watch_job(self, job_id: int, websocket: WebSocket):
        """İş takibine başla"""
        if job_id not in self.job_watchers:
            self.job_watchers[job_id] = set()
        self.job_watchers[job_id].add(websocket)
    
    def unwatch_job(self, job_id: int, websocket: WebSocket):
        """İş takibini durdur"""
        if job_id in self.job_watchers:
            self.job_watchers[job_id].discard(websocket)
            if not self.job_watchers[job_id]:
                del self.job_watchers[job_id]
    
    async def notify_job_update(self, job_id: int, job_data: Dict[str, Any]):
        """İş güncellemesini bildir"""
        if job_id in self.job_watchers:
            message = {
                "type": "job_update",
                "job_id": job_id,
                "data": job_data,
                "timestamp": datetime.now().isoformat()
            }
            
            disconnected = set()
            for websocket in self.job_watchers[job_id]:
                try:
                    await websocket.send_text(json.dumps(message, ensure_ascii=False))
                except Exception as e:
                    logging.error(f"İş güncelleme bildirimi hatası: {e}")
                    disconnected.add(websocket)
            
            # Bağlantısı kesilen websocket'leri temizle
            for websocket in disconnected:
                self.job_watchers[job_id].discard(websocket)
    
    async def notify_system_status(self, status_data: Dict[str, Any]):
        """Sistem durumu bildirimi"""
        message = {
            "type": "system_status",
            "data": status_data,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.broadcast(message)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Bağlantı istatistiklerini al"""
        return {
            "total_clients": len(self.active_connections),
            "total_users": len(self.user_connections),
            "watched_jobs": len(self.job_watchers),
            "active_connections": sum(len(conns) for conns in self.active_connections.values()),
            "user_connections": sum(len(conns) for conns in self.user_connections.values())
        }

class WebSocketHandler:
    """WebSocket mesaj işleyicisi"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
    
    async def handle_message(self, websocket: WebSocket, message: str, client_id: str, user_id: str = None):
        """Gelen mesajı işle"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "watch_job":
                job_id = data.get("job_id")
                if job_id:
                    self.connection_manager.watch_job(job_id, websocket)
                    await self.connection_manager.send_personal_message({
                        "type": "job_watch_started",
                        "job_id": job_id,
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
            
            elif message_type == "unwatch_job":
                job_id = data.get("job_id")
                if job_id:
                    self.connection_manager.unwatch_job(job_id, websocket)
                    await self.connection_manager.send_personal_message({
                        "type": "job_watch_stopped",
                        "job_id": job_id,
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
            
            elif message_type == "ping":
                await self.connection_manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }, websocket)
            
            elif message_type == "get_stats":
                stats = self.connection_manager.get_connection_stats()
                await self.connection_manager.send_personal_message({
                    "type": "connection_stats",
                    "data": stats,
                    "timestamp": datetime.now().isoformat()
                }, websocket)
            
            else:
                await self.connection_manager.send_personal_message({
                    "type": "error",
                    "message": f"Bilinmeyen mesaj türü: {message_type}",
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
        except json.JSONDecodeError:
            await self.connection_manager.send_personal_message({
                "type": "error",
                "message": "Geçersiz JSON formatı",
                "timestamp": datetime.now().isoformat()
            }, websocket)
        except Exception as e:
            logging.error(f"Mesaj işleme hatası: {e}")
            await self.connection_manager.send_personal_message({
                "type": "error",
                "message": f"Mesaj işleme hatası: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, websocket)

class RealTimeJobTracker:
    """Gerçek zamanlı iş takipçisi"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.tracking_jobs = set()
        self.tracking_task = None
    
    async def start_tracking(self):
        """İş takibini başlat"""
        if self.tracking_task is None:
            self.tracking_task = asyncio.create_task(self._track_jobs())
    
    async def stop_tracking(self):
        """İş takibini durdur"""
        if self.tracking_task:
            self.tracking_task.cancel()
            try:
                await self.tracking_task
            except asyncio.CancelledError:
                pass
            self.tracking_task = None
    
    async def add_job_to_tracking(self, job_id: int):
        """İş takibine ekle"""
        self.tracking_jobs.add(job_id)
        await self.start_tracking()
    
    async def remove_job_from_tracking(self, job_id: int):
        """İş takibinden çıkar"""
        self.tracking_jobs.discard(job_id)
        if not self.tracking_jobs and self.tracking_task:
            await self.stop_tracking()
    
    async def _track_jobs(self):
        """İş takibi döngüsü"""
        from core.database import get_db_manager
        
        db_manager = get_db_manager()
        
        while True:
            try:
                for job_id in list(self.tracking_jobs):
                    job = db_manager.get_job(job_id)
                    if job:
                        # İş durumu değiştiyse bildir
                        job_data = {
                            "id": job.id,
                            "status": job.status,
                            "progress": job.progress,
                            "current_step": job.current_step,
                            "updated_at": job.updated_at.isoformat() if job.updated_at else None
                        }
                        
                        await self.connection_manager.notify_job_update(job_id, job_data)
                        
                        # Tamamlanan işleri takipten çıkar
                        if job.status in ['completed', 'failed', 'cancelled']:
                            await self.remove_job_from_tracking(job_id)
                    else:
                        # İş bulunamadıysa takipten çıkar
                        await self.remove_job_from_tracking(job_id)
                
                # 2 saniye bekle
                await asyncio.sleep(2)
                
            except Exception as e:
                logging.error(f"İş takibi hatası: {e}")
                await asyncio.sleep(5)  # Hata durumunda daha uzun bekle

# Global instances
connection_manager = ConnectionManager()
websocket_handler = WebSocketHandler(connection_manager)
job_tracker = RealTimeJobTracker(connection_manager)

def get_connection_manager() -> ConnectionManager:
    """Connection manager'ı al"""
    return connection_manager

def get_websocket_handler() -> WebSocketHandler:
    """WebSocket handler'ı al"""
    return websocket_handler

def get_job_tracker() -> RealTimeJobTracker:
    """Job tracker'ı al"""
    return job_tracker
