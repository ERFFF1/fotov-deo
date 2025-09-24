#!/usr/bin/env python3
"""
Error Manager
Çok aşamalı hata yönetimi ve retry mekanizması
"""

import time
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum

class ErrorType(Enum):
    """Hata türleri"""
    CRITICAL = "critical"      # Sistem durdurucu
    RETRYABLE = "retryable"    # Yeniden denenebilir
    SKIPPABLE = "skippable"    # Atlanabilir
    USER_ERROR = "user_error"  # Kullanıcı hatası

class RetryStrategy(Enum):
    """Retry stratejileri"""
    IMMEDIATE = "immediate"    # Hemen tekrar dene
    EXPONENTIAL = "exponential" # Üstel artış
    LINEAR = "linear"          # Doğrusal artış
    FIXED = "fixed"           # Sabit aralık

class ErrorManager:
    """Hata yönetimi ve retry mekanizması"""
    
    def __init__(self):
        self.setup_directories()
        self.error_log = []
        self.retry_configs = self._get_default_retry_configs()
        self.error_handlers = self._get_default_error_handlers()
        
    def setup_directories(self):
        """Gerekli klasörleri oluştur"""
        directories = [
            "storage/logs/errors",
            "storage/artifacts/errors"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _get_default_retry_configs(self) -> Dict[str, Dict[str, Any]]:
        """Varsayılan retry konfigürasyonları"""
        return {
            "face_detect": {
                "max_retries": 3,
                "strategy": RetryStrategy.EXPONENTIAL,
                "base_delay": 1.0,
                "max_delay": 10.0,
                "error_types": [ErrorType.RETRYABLE]
            },
            "face_swap": {
                "max_retries": 2,
                "strategy": RetryStrategy.LINEAR,
                "base_delay": 2.0,
                "max_delay": 8.0,
                "error_types": [ErrorType.RETRYABLE]
            },
            "video_process": {
                "max_retries": 1,
                "strategy": RetryStrategy.FIXED,
                "base_delay": 5.0,
                "max_delay": 5.0,
                "error_types": [ErrorType.RETRYABLE]
            },
            "model_load": {
                "max_retries": 5,
                "strategy": RetryStrategy.EXPONENTIAL,
                "base_delay": 0.5,
                "max_delay": 15.0,
                "error_types": [ErrorType.RETRYABLE]
            },
            "file_operation": {
                "max_retries": 3,
                "strategy": RetryStrategy.LINEAR,
                "base_delay": 1.0,
                "max_delay": 5.0,
                "error_types": [ErrorType.RETRYABLE, ErrorType.SKIPPABLE]
            }
        }
    
    def _get_default_error_handlers(self) -> Dict[ErrorType, Callable]:
        """Varsayılan hata işleyicileri"""
        return {
            ErrorType.CRITICAL: self._handle_critical_error,
            ErrorType.RETRYABLE: self._handle_retryable_error,
            ErrorType.SKIPPABLE: self._handle_skippable_error,
            ErrorType.USER_ERROR: self._handle_user_error
        }
    
    def execute_with_retry(self, 
                          operation: Callable,
                          operation_name: str,
                          *args,
                          **kwargs) -> Dict[str, Any]:
        """Retry mekanizması ile işlem çalıştır"""
        retry_config = self.retry_configs.get(operation_name, self.retry_configs["file_operation"])
        max_retries = retry_config["max_retries"]
        strategy = retry_config["strategy"]
        base_delay = retry_config["base_delay"]
        max_delay = retry_config["max_delay"]
        
        last_error = None
        attempt = 0
        
        while attempt <= max_retries:
            try:
                # İşlemi çalıştır
                result = operation(*args, **kwargs)
                
                # Başarılı ise logla ve döndür
                if attempt > 0:
                    self._log_retry_success(operation_name, attempt)
                
                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt + 1,
                    "operation": operation_name
                }
                
            except Exception as e:
                last_error = e
                attempt += 1
                
                # Hata türünü belirle
                error_type = self._classify_error(e)
                
                # Hata işleyicisini çalıştır
                handler_result = self._handle_error(e, error_type, operation_name, attempt)
                
                # Kritik hata ise dur
                if error_type == ErrorType.CRITICAL:
                    break
                
                # Son deneme ise dur
                if attempt > max_retries:
                    break
                
                # Atlanabilir hata ise dur
                if error_type == ErrorType.SKIPPABLE and handler_result.get("skip", False):
                    break
                
                # Retry gecikmesi hesapla
                delay = self._calculate_delay(strategy, base_delay, max_delay, attempt)
                
                # Retry logla
                self._log_retry_attempt(operation_name, attempt, e, delay)
                
                # Gecikme
                time.sleep(delay)
        
        # Tüm denemeler başarısız
        return {
            "success": False,
            "error": str(last_error),
            "error_type": self._classify_error(last_error).value,
            "attempts": attempt,
            "operation": operation_name
        }
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Hata türünü sınıflandır"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Kritik hatalar
        if any(keyword in error_str for keyword in [
            "memory", "disk space", "permission denied", "file not found"
        ]):
            return ErrorType.CRITICAL
        
        # Kullanıcı hataları
        if any(keyword in error_str for keyword in [
            "invalid input", "unsupported format", "file too large"
        ]):
            return ErrorType.USER_ERROR
        
        # Atlanabilir hatalar
        if any(keyword in error_str for keyword in [
            "timeout", "connection", "network"
        ]):
            return ErrorType.SKIPPABLE
        
        # Varsayılan: yeniden denenebilir
        return ErrorType.RETRYABLE
    
    def _handle_critical_error(self, error: Exception, operation: str, attempt: int) -> Dict[str, Any]:
        """Kritik hata işleme"""
        self._log_error(error, ErrorType.CRITICAL, operation, attempt)
        return {"stop": True, "message": "Kritik hata: İşlem durduruldu"}
    
    def _handle_retryable_error(self, error: Exception, operation: str, attempt: int) -> Dict[str, Any]:
        """Yeniden denenebilir hata işleme"""
        self._log_error(error, ErrorType.RETRYABLE, operation, attempt)
        return {"retry": True, "message": "Hata: Yeniden deneniyor"}
    
    def _handle_skippable_error(self, error: Exception, operation: str, attempt: int) -> Dict[str, Any]:
        """Atlanabilir hata işleme"""
        self._log_error(error, ErrorType.SKIPPABLE, operation, attempt)
        return {"skip": True, "message": "Hata: İşlem atlanıyor"}
    
    def _handle_user_error(self, error: Exception, operation: str, attempt: int) -> Dict[str, Any]:
        """Kullanıcı hatası işleme"""
        self._log_error(error, ErrorType.USER_ERROR, operation, attempt)
        return {"stop": True, "message": "Kullanıcı hatası: Lütfen girdiyi kontrol edin"}
    
    def _handle_error(self, error: Exception, error_type: ErrorType, operation: str, attempt: int) -> Dict[str, Any]:
        """Hata işleme"""
        handler = self.error_handlers.get(error_type, self._handle_retryable_error)
        return handler(error, operation, attempt)
    
    def _calculate_delay(self, strategy: RetryStrategy, base_delay: float, max_delay: float, attempt: int) -> float:
        """Retry gecikmesi hesapla"""
        if strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif strategy == RetryStrategy.FIXED:
            return min(base_delay, max_delay)
        elif strategy == RetryStrategy.LINEAR:
            return min(base_delay * attempt, max_delay)
        elif strategy == RetryStrategy.EXPONENTIAL:
            return min(base_delay * (2 ** (attempt - 1)), max_delay)
        else:
            return base_delay
    
    def _log_error(self, error: Exception, error_type: ErrorType, operation: str, attempt: int):
        """Hata logla"""
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type.value,
            "operation": operation,
            "attempt": attempt,
            "error_message": str(error),
            "error_class": type(error).__name__,
            "traceback": traceback.format_exc()
        }
        
        self.error_log.append(error_data)
        self._save_error_log(error_data)
    
    def _log_retry_attempt(self, operation: str, attempt: int, error: Exception, delay: float):
        """Retry denemesi logla"""
        retry_data = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "attempt": attempt,
            "error": str(error),
            "delay": delay,
            "type": "retry_attempt"
        }
        
        self._save_error_log(retry_data)
    
    def _log_retry_success(self, operation: str, attempts: int):
        """Retry başarısı logla"""
        success_data = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "attempts": attempts,
            "type": "retry_success"
        }
        
        self._save_error_log(success_data)
    
    def _save_error_log(self, log_data: Dict[str, Any]):
        """Hata logunu dosyaya kaydet"""
        try:
            log_file = Path("storage/logs/errors/error_log.jsonl")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")
                
        except Exception as e:
            print(f"Hata log kaydetme hatası: {e}")
    
    def get_error_summary(self, operation: str = None) -> Dict[str, Any]:
        """Hata özeti döndür"""
        if operation:
            filtered_logs = [log for log in self.error_log if log.get("operation") == operation]
        else:
            filtered_logs = self.error_log
        
        if not filtered_logs:
            return {"total_errors": 0, "operations": []}
        
        # Hata türlerine göre grupla
        error_types = {}
        operations = {}
        
        for log in filtered_logs:
            error_type = log.get("error_type", "unknown")
            op = log.get("operation", "unknown")
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            operations[op] = operations.get(op, 0) + 1
        
        return {
            "total_errors": len(filtered_logs),
            "error_types": error_types,
            "operations": operations,
            "recent_errors": filtered_logs[-10:] if len(filtered_logs) > 10 else filtered_logs
        }
    
    def clear_error_log(self):
        """Hata logunu temizle"""
        self.error_log.clear()
        try:
            log_file = Path("storage/logs/errors/error_log.jsonl")
            if log_file.exists():
                log_file.unlink()
        except Exception as e:
            print(f"Hata log temizleme hatası: {e}")
    
    def add_retry_config(self, operation: str, config: Dict[str, Any]):
        """Retry konfigürasyonu ekle"""
        self.retry_configs[operation] = config
    
    def add_error_handler(self, error_type: ErrorType, handler: Callable):
        """Hata işleyici ekle"""
        self.error_handlers[error_type] = handler

# Global instance
_error_manager = None

def get_error_manager() -> ErrorManager:
    """Global error manager instance döndür"""
    global _error_manager
    if _error_manager is None:
        _error_manager = ErrorManager()
    return _error_manager

# Decorator for automatic error handling
def with_error_handling(operation_name: str, retry_config: Dict[str, Any] = None):
    """Hata yönetimi decorator'ı"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            error_manager = get_error_manager()
            
            # Retry konfigürasyonu ekle
            if retry_config:
                error_manager.add_retry_config(operation_name, retry_config)
            
            # İşlemi retry ile çalıştır
            return error_manager.execute_with_retry(func, operation_name, *args, **kwargs)
        
        return wrapper
    return decorator
