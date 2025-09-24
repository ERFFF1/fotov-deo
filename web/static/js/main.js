/**
 * Face/Video AI Studio - Modern JavaScript
 * Drag & Drop, Real-time Updates, Interactive UI
 */

class AIStudio {
    constructor() {
        this.uploadedFiles = new Map();
        this.activeJobs = new Set();
        this.updateInterval = null;
        this.init();
    }

    init() {
        this.setupDragAndDrop();
        this.setupFileUpload();
        this.setupJobUpdates();
        this.setupModals();
        this.setupForms();
        console.log('🎯 AI Studio initialized');
    }

    // Drag & Drop Functionality
    setupDragAndDrop() {
        const dropZones = document.querySelectorAll('.drop-zone');
        
        dropZones.forEach(zone => {
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('dragover');
            });

            zone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
            });

            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
                
                const files = Array.from(e.dataTransfer.files);
                this.handleFiles(files, zone);
            });

            zone.addEventListener('click', () => {
                const input = zone.querySelector('input[type="file"]');
                if (input) input.click();
            });
        });
    }

    // File Upload Handling
    setupFileUpload() {
        const fileInputs = document.querySelectorAll('input[type="file"]');
        
        fileInputs.forEach(input => {
            input.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                this.handleFiles(files, input.closest('.drop-zone'));
            });
        });
    }

    async handleFiles(files, container) {
        const fileType = container.dataset.fileType || 'image';
        
        for (const file of files) {
            if (this.validateFile(file, fileType)) {
                await this.uploadFile(file, container);
            }
        }
    }

    validateFile(file, expectedType) {
        const maxSize = 100 * 1024 * 1024; // 100MB
        const allowedTypes = {
            image: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
            video: ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
            audio: ['.wav', '.mp3', '.aac', '.flac']
        };

        if (file.size > maxSize) {
            this.showAlert('Dosya çok büyük (max 100MB)', 'danger');
            return false;
        }

        const extension = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowedTypes[expectedType].includes(extension)) {
            this.showAlert(`Desteklenmeyen dosya formatı: ${extension}`, 'danger');
            return false;
        }

        return true;
    }

    async uploadFile(file, container) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            this.showLoading(container);
            
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.success) {
                this.uploadedFiles.set(file.name, result);
                this.showFilePreview(file, result, container);
                this.showAlert('Dosya başarıyla yüklendi', 'success');
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showAlert('Dosya yükleme hatası: ' + error.message, 'danger');
        } finally {
            this.hideLoading(container);
        }
    }

    showFilePreview(file, uploadResult, container) {
        const preview = document.createElement('div');
        preview.className = 'file-preview';
        preview.dataset.filename = file.name;
        
        const isImage = file.type.startsWith('image/');
        const isVideo = file.type.startsWith('video/');
        
        let previewHTML = `
            <div class="file-preview-info">
                <div class="file-preview-name">${file.name}</div>
                <div class="file-preview-size">${this.formatFileSize(file.size)}</div>
            </div>
            <div class="file-preview-remove" onclick="aiStudio.removeFile('${file.name}')">
                <i class="fas fa-times"></i>
            </div>
        `;

        if (isImage) {
            const img = document.createElement('img');
            img.className = 'file-preview-image';
            img.src = URL.createObjectURL(file);
            preview.appendChild(img);
        } else if (isVideo) {
            const video = document.createElement('video');
            video.className = 'file-preview-image';
            video.src = URL.createObjectURL(file);
            video.muted = true;
            preview.appendChild(video);
        }

        preview.innerHTML += previewHTML;
        
        // Container'daki mevcut preview'ları temizle
        const existingPreviews = container.querySelectorAll('.file-preview');
        existingPreviews.forEach(p => p.remove());
        
        container.appendChild(preview);
        
        // Drop zone'u gizle
        const dropZone = container.querySelector('.drop-zone');
        if (dropZone) {
            dropZone.style.display = 'none';
        }
    }

    removeFile(filename) {
        this.uploadedFiles.delete(filename);
        
        const preview = document.querySelector(`[data-filename="${filename}"]`);
        if (preview) {
            preview.remove();
        }
        
        // Drop zone'u tekrar göster
        const container = preview?.closest('.file-container');
        const dropZone = container?.querySelector('.drop-zone');
        if (dropZone) {
            dropZone.style.display = 'block';
        }
    }

    // Job Management
    setupJobUpdates() {
        // Her 3 saniyede bir aktif işleri güncelle
        this.updateInterval = setInterval(() => {
            this.updateActiveJobs();
        }, 3000);
    }

    async updateActiveJobs() {
        const activeJobElements = document.querySelectorAll('[data-job-id]');
        
        for (const element of activeJobElements) {
            const jobId = element.dataset.jobId;
            if (jobId && !this.activeJobs.has(jobId)) {
                this.activeJobs.add(jobId);
                await this.updateJobStatus(jobId);
            }
        }
    }

    async updateJobStatus(jobId) {
        try {
            const response = await fetch(`/api/jobs/${jobId}`);
            const job = await response.json();
            
            this.updateJobElement(job);
            
            // İş tamamlandıysa aktif listeden çıkar
            if (['completed', 'failed', 'cancelled'].includes(job.status)) {
                this.activeJobs.delete(jobId);
            }
        } catch (error) {
            console.error('Job update error:', error);
        }
    }

    updateJobElement(job) {
        const element = document.querySelector(`[data-job-id="${job.id}"]`);
        if (!element) return;

        // Durum güncelle
        const statusElement = element.querySelector('.status');
        if (statusElement) {
            statusElement.className = `status ${job.status}`;
            statusElement.textContent = job.status;
        }

        // İlerleme güncelle
        const progressElement = element.querySelector('.progress-bar');
        if (progressElement) {
            progressElement.style.width = `${job.progress}%`;
        }

        // Mevcut adım güncelle
        const stepElement = element.querySelector('.current-step');
        if (stepElement && job.current_step) {
            stepElement.textContent = job.current_step;
        }

        // Tamamlanan işler için indirme linki ekle
        if (job.status === 'completed' && job.output_path) {
            const actionsElement = element.querySelector('.job-actions');
            if (actionsElement && !actionsElement.querySelector('.download-btn')) {
                const downloadBtn = document.createElement('a');
                downloadBtn.className = 'btn btn-success download-btn';
                downloadBtn.href = `/api/jobs/${job.id}/result`;
                downloadBtn.innerHTML = '<i class="fas fa-download"></i> İndir';
                actionsElement.appendChild(downloadBtn);
            }
        }
    }

    // Job Creation
    async createJob(jobData) {
        try {
            const response = await fetch('/api/jobs', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(jobData)
            });

            const result = await response.json();
            
            if (response.ok) {
                this.showAlert('İş başarıyla oluşturuldu', 'success');
                this.activeJobs.add(result.id);
                
                // Job detail sayfasına yönlendir
                setTimeout(() => {
                    window.location.href = `/job/${result.id}`;
                }, 1000);
                
                return result;
            } else {
                throw new Error(result.detail || 'Job creation failed');
            }
        } catch (error) {
            console.error('Job creation error:', error);
            this.showAlert('İş oluşturma hatası: ' + error.message, 'danger');
            throw error;
        }
    }

    // Modal Management
    setupModals() {
        // Modal açma
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-modal]')) {
                const modalId = e.target.dataset.modal;
                this.openModal(modalId);
            }
        });

        // Modal kapatma
        document.addEventListener('click', (e) => {
            if (e.target.matches('.modal-close, .modal')) {
                this.closeModal();
            }
        });

        // ESC ile modal kapatma
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
    }

    openModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('show');
            document.body.style.overflow = 'hidden';
        }
    }

    closeModal() {
        const modal = document.querySelector('.modal.show');
        if (modal) {
            modal.classList.remove('show');
            document.body.style.overflow = '';
        }
    }

    // Form Management
    setupForms() {
        // Preset değişikliklerini dinle
        const presetSelects = document.querySelectorAll('select[name="preset"]');
        presetSelects.forEach(select => {
            select.addEventListener('change', (e) => {
                this.loadPreset(e.target.value);
            });
        });

        // Form gönderimlerini dinle
        const forms = document.querySelectorAll('form[data-job-type]');
        forms.forEach(form => {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleFormSubmit(form);
            });
        });
    }

    async loadPreset(presetName) {
        if (!presetName) return;

        try {
            const response = await fetch('/api/presets');
            const data = await response.json();
            
            const preset = data.presets.find(p => p.name === presetName);
            if (preset) {
                this.applyPreset(preset);
            }
        } catch (error) {
            console.error('Preset loading error:', error);
        }
    }

    applyPreset(preset) {
        // Preset parametrelerini form alanlarına uygula
        const form = document.querySelector('form[data-job-type]');
        if (!form) return;

        // Job type'ı güncelle
        const jobTypeInput = form.querySelector('input[name="job_type"]');
        if (jobTypeInput) {
            jobTypeInput.value = preset.job_type;
        }

        // Parametreleri güncelle
        Object.entries(preset.params).forEach(([key, value]) => {
            const input = form.querySelector(`[name="params.${key}"]`);
            if (input) {
                if (input.type === 'checkbox') {
                    input.checked = value;
                } else {
                    input.value = value;
                }
            }
        });
    }

    async handleFormSubmit(form) {
        const jobType = form.dataset.jobType;
        const formData = new FormData(form);
        
        // Form verilerini topla
        const jobData = {
            job_type: jobType,
            inputs: {},
            params: {},
            consent_tag: formData.get('consent_tag') || 'unknown'
        };

        // Input dosyalarını topla
        const inputFields = form.querySelectorAll('[data-input-type]');
        inputFields.forEach(field => {
            const inputType = field.dataset.inputType;
            const filename = field.dataset.filename;
            if (filename && this.uploadedFiles.has(filename)) {
                jobData.inputs[inputType] = this.uploadedFiles.get(filename).filepath;
            }
        });

        // Parametreleri topla
        const paramFields = form.querySelectorAll('[name^="params."]');
        paramFields.forEach(field => {
            const paramName = field.name.replace('params.', '');
            if (field.type === 'checkbox') {
                jobData.params[paramName] = field.checked;
            } else {
                jobData.params[paramName] = field.value;
            }
        });

        // Preset varsa ekle
        const preset = formData.get('preset');
        if (preset) {
            jobData.preset = preset;
        }

        // Rıza kontrolü
        if (jobData.consent_tag === 'unknown') {
            this.showAlert('Lütfen rıza durumunu belirtin', 'warning');
            return;
        }

        try {
            await this.createJob(jobData);
        } catch (error) {
            // Hata zaten createJob'da gösterildi
        }
    }

    // Utility Functions
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    showAlert(message, type = 'info') {
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; font-size: 1.2rem; cursor: pointer;">&times;</button>
            </div>
        `;

        // Sayfanın üstüne ekle
        const container = document.querySelector('.container');
        if (container) {
            container.insertBefore(alert, container.firstChild);
        }

        // 5 saniye sonra otomatik kaldır
        setTimeout(() => {
            if (alert.parentElement) {
                alert.remove();
            }
        }, 5000);
    }

    showLoading(container) {
        const loading = document.createElement('div');
        loading.className = 'loading';
        loading.innerHTML = '<div class="spinner"></div> Yükleniyor...';
        container.appendChild(loading);
    }

    hideLoading(container) {
        const loading = container.querySelector('.loading');
        if (loading) {
            loading.remove();
        }
    }

    // Cleanup
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }
}

// Global instance
let aiStudio;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    aiStudio = new AIStudio();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (aiStudio) {
        aiStudio.destroy();
    }
});
