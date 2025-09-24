#!/usr/bin/env python3
"""
Veritabanı Yönetimi
SQLAlchemy tabanlı veritabanı şeması ve işlemleri
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from pathlib import Path

# Veritabanı taban sınıfı
Base = declarative_base()

class Job(Base):
    """İş tablosu"""
    __tablename__ = 'jobs'
    
    id = Column(Integer, primary_key=True, index=True)
    job_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), default='pending', index=True)  # pending, running, completed, failed, cancelled
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Konfigürasyon
    yaml_config = Column(Text)  # YAML konfigürasyon
    model_versions = Column(JSON)  # Kullanılan model versiyonları
    
    # İlerleme ve metrikler
    progress = Column(Float, default=0.0)
    current_step = Column(String(100))
    total_steps = Column(Integer, default=0)
    completed_steps = Column(Integer, default=0)
    
    # Sonuçlar
    output_path = Column(String(500))
    error_message = Column(Text)
    metrics_json = Column(JSON)  # İşlem metrikleri
    
    # Rıza ve etik
    consent_tag = Column(String(20), default='unknown')
    user_id = Column(String(100))  # Kullanıcı kimliği (opsiyonel)

class Artifact(Base):
    """Artefakt tablosu (ara çıktılar)"""
    __tablename__ = 'artifacts'
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, nullable=False, index=True)
    kind = Column(String(50), nullable=False)  # frame, mask, enhanced, etc.
    path = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    meta_json = Column(JSON)  # Ek meta veriler

class Model(Base):
    """Model tablosu"""
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    sha256 = Column(String(64), nullable=False, unique=True)
    path = Column(String(500), nullable=False)
    source_url = Column(String(500))
    size_mb = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class Preset(Base):
    """Preset tablosu"""
    __tablename__ = 'presets'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text)
    yaml_config = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_public = Column(Boolean, default=True)
    user_id = Column(String(100))  # Preset sahibi

class Consent(Base):
    """Rıza tablosu"""
    __tablename__ = 'consents'
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, nullable=False, index=True)
    person_id = Column(String(100))  # Kişi kimliği (hash)
    tag = Column(String(20), nullable=False)  # consented, demo, unknown
    evidence_path = Column(String(500))  # Rıza kanıtı dosyası
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Veritabanı yöneticisi"""
    
    def __init__(self, db_url: str = "sqlite:///db/app.db"):
        self.db_url = db_url
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Veritabanı dizinini oluştur
        if db_url.startswith("sqlite"):
            db_path = Path(db_url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Tabloları oluştur
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Veritabanı oturumu al"""
        return self.SessionLocal()
    
    def create_job(self, job_type: str, yaml_config: str, consent_tag: str = 'unknown', 
                   user_id: str = None) -> Job:
        """Yeni iş oluştur"""
        with self.get_session() as session:
            job = Job(
                job_type=job_type,
                yaml_config=yaml_config,
                consent_tag=consent_tag,
                user_id=user_id
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            return job
    
    def get_job(self, job_id: int) -> Optional[Job]:
        """İş bilgisini al"""
        with self.get_session() as session:
            return session.query(Job).filter(Job.id == job_id).first()
    
    def update_job_status(self, job_id: int, status: str, progress: float = None, 
                         current_step: str = None, error_message: str = None) -> bool:
        """İş durumunu güncelle"""
        with self.get_session() as session:
            job = session.query(Job).filter(Job.id == job_id).first()
            if not job:
                return False
            
            job.status = status
            if progress is not None:
                job.progress = progress
            if current_step is not None:
                job.current_step = current_step
            if error_message is not None:
                job.error_message = error_message
            
            job.updated_at = datetime.utcnow()
            session.commit()
            return True
    
    def update_job_progress(self, job_id: int, completed_steps: int, total_steps: int = None) -> bool:
        """İş ilerlemesini güncelle"""
        with self.get_session() as session:
            job = session.query(Job).filter(Job.id == job_id).first()
            if not job:
                return False
            
            job.completed_steps = completed_steps
            if total_steps is not None:
                job.total_steps = total_steps
            
            if job.total_steps > 0:
                job.progress = (completed_steps / job.total_steps) * 100
            
            job.updated_at = datetime.utcnow()
            session.commit()
            return True
    
    def set_job_output(self, job_id: int, output_path: str, metrics: Dict[str, Any] = None) -> bool:
        """İş çıktısını ayarla"""
        with self.get_session() as session:
            job = session.query(Job).filter(Job.id == job_id).first()
            if not job:
                return False
            
            job.output_path = output_path
            if metrics:
                job.metrics_json = metrics
            job.status = 'completed'
            job.progress = 100.0
            job.updated_at = datetime.utcnow()
            session.commit()
            return True
    
    def list_jobs(self, limit: int = 50, offset: int = 0, status: str = None) -> List[Job]:
        """İşleri listele"""
        with self.get_session() as session:
            query = session.query(Job)
            if status:
                query = query.filter(Job.status == status)
            return query.order_by(Job.created_at.desc()).offset(offset).limit(limit).all()
    
    def create_artifact(self, job_id: int, kind: str, path: str, meta: Dict[str, Any] = None) -> Artifact:
        """Artefakt oluştur"""
        with self.get_session() as session:
            artifact = Artifact(
                job_id=job_id,
                kind=kind,
                path=path,
                meta_json=meta
            )
            session.add(artifact)
            session.commit()
            session.refresh(artifact)
            return artifact
    
    def get_job_artifacts(self, job_id: int) -> List[Artifact]:
        """İş artefaktlarını al"""
        with self.get_session() as session:
            return session.query(Artifact).filter(Artifact.job_id == job_id).all()
    
    def register_model(self, name: str, version: str, sha256: str, path: str, 
                      source_url: str = None, size_mb: float = None) -> Model:
        """Model kaydet"""
        with self.get_session() as session:
            # Mevcut modeli kontrol et
            existing = session.query(Model).filter(
                Model.name == name, 
                Model.version == version
            ).first()
            
            if existing:
                # Güncelle
                existing.sha256 = sha256
                existing.path = path
                existing.source_url = source_url
                existing.size_mb = size_mb
                existing.last_updated = datetime.utcnow()
                session.commit()
                return existing
            else:
                # Yeni model
                model = Model(
                    name=name,
                    version=version,
                    sha256=sha256,
                    path=path,
                    source_url=source_url,
                    size_mb=size_mb
                )
                session.add(model)
                session.commit()
                session.refresh(model)
                return model
    
    def get_model(self, name: str, version: str = None) -> Optional[Model]:
        """Model bilgisini al"""
        with self.get_session() as session:
            query = session.query(Model).filter(Model.name == name, Model.is_active == True)
            if version:
                query = query.filter(Model.version == version)
            return query.first()
    
    def list_models(self) -> List[Model]:
        """Modelleri listele"""
        with self.get_session() as session:
            return session.query(Model).filter(Model.is_active == True).all()

# Global veritabanı yöneticisi
db_manager = DatabaseManager()

def get_db_manager() -> DatabaseManager:
    """Veritabanı yöneticisini al"""
    return db_manager
