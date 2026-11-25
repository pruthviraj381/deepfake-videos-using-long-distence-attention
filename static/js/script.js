/**
 * Deepfake Video Detection using Long-Distance Attention
 * Frontend JavaScript
 */

document.addEventListener('DOMContentLoaded', function () {
    // 1. Mobile Sidebar Toggle
    const sidebarToggle = document.getElementById('sidebarToggle');
    const appSidebar = document.querySelector('.app-sidebar');

    if (sidebarToggle && appSidebar) {
        sidebarToggle.addEventListener('click', function () {
            appSidebar.classList.toggle('show');
        });
    }

    // 2. Drag & Drop Video Upload
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('video_file');
    const filePreview = document.getElementById('filePreview');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const fileSizeDisplay = document.getElementById('fileSizeDisplay');
    const uploadForm = document.getElementById('videoUploadForm');
    const removeFileBtn = document.getElementById('removeFileBtn');

    if (dropzone && fileInput) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelection(files[0]);
            }
        });

        dropzone.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleFileSelection(fileInput.files[0]);
            }
        });

        if (removeFileBtn) {
            removeFileBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                fileInput.value = '';
                if (filePreview) filePreview.style.display = 'none';
            });
        }
    }

    function handleFileSelection(file) {
        const validExtensions = ['mp4', 'avi', 'mov'];
        const ext = file.name.split('.').pop().toLowerCase();

        if (!validExtensions.includes(ext)) {
            alert('Invalid file format. Please upload an MP4, AVI, or MOV video.');
            fileInput.value = '';
            if (filePreview) filePreview.style.display = 'none';
            return;
        }

        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        if (file.size > 50 * 1024 * 1024) {
            alert(`File size (${sizeMB} MB) exceeds the 50 MB limit.`);
            fileInput.value = '';
            if (filePreview) filePreview.style.display = 'none';
            return;
        }

        if (fileNameDisplay) fileNameDisplay.textContent = file.name;
        if (fileSizeDisplay) fileSizeDisplay.textContent = `${sizeMB} MB`;
        if (filePreview) filePreview.style.display = 'flex';
    }

    // 3. Analysis Processing Modal Animation
    const processingModal = document.getElementById('processingModal');
    if (uploadForm && processingModal) {
        uploadForm.addEventListener('submit', function (e) {
            if (!fileInput || !fileInput.files.length) {
                alert('Please select or drop a video file first.');
                e.preventDefault();
                return;
            }

            // Display modal
            processingModal.style.display = 'flex';

            // Animated step simulator
            const steps = [
                { id: 'step-1', delay: 400 },
                { id: 'step-2', delay: 1800 },
                { id: 'step-3', delay: 3500 },
                { id: 'step-4', delay: 5200 },
                { id: 'step-5', delay: 7000 }
            ];

            steps.forEach(s => {
                setTimeout(() => {
                    const el = document.getElementById(s.id);
                    if (el) {
                        el.classList.add('active');
                        el.innerHTML = `<i class="bi bi-check-circle-fill text-success me-2"></i>` + el.textContent.trim();
                    }
                }, s.delay);
            });
        });
    }

    // 5. Admin Training Live Polling & Controls
    const startTrainingBtn = document.getElementById('startTrainingBtn');
    const trainingProgressBar = document.getElementById('trainingProgressBar');
    const trainingProgressText = document.getElementById('trainingProgressText');
    const trainingStatusMessage = document.getElementById('trainingStatusMessage');
    const epochHistoryTable = document.getElementById('epochHistoryTableBody');

    let pollingInterval = null;

    if (startTrainingBtn) {
        startTrainingBtn.addEventListener('click', function () {
            const epochs = document.getElementById('trainEpochs') ? document.getElementById('trainEpochs').value : 5;
            const batchSize = document.getElementById('trainBatchSize') ? document.getElementById('trainBatchSize').value : 2;
            const lr = document.getElementById('trainLr') ? document.getElementById('trainLr').value : 0.0003;

            startTrainingBtn.disabled = true;
            startTrainingBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Training in Progress...`;

            if (trainingStatusMessage) trainingStatusMessage.textContent = 'Initializing training run on CPU...';

            // Send training request
            const formData = new FormData();
            formData.append('epochs', epochs);
            formData.append('batch_size', batchSize);
            formData.append('lr', lr);

            fetch('/admin/api/train/start', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    pollTrainingStatus();
                } else {
                    alert('Error starting training: ' + data.message);
                    resetTrainingButton();
                }
            })
            .catch(err => {
                console.error(err);
                alert('Request failed to start training.');
                resetTrainingButton();
            });
        });
    }

    function pollTrainingStatus() {
        if (pollingInterval) clearInterval(pollingInterval);

        pollingInterval = setInterval(() => {
            fetch('/admin/api/train/status')
                .then(res => res.json())
                .then(status => {
                    updateTrainingUI(status);
                    if (!status.is_training) {
                        clearInterval(pollingInterval);
                        resetTrainingButton();
                        if (status.error) {
                            alert('Training stopped with error: ' + status.error);
                        } else {
                            setTimeout(() => {
                                alert('Model training completed! Best weights saved.');
                                window.location.reload();
                            }, 800);
                        }
                    }
                })
                .catch(err => console.error('Status polling error:', err));
        }, 1500);
    }

    function updateTrainingUI(status) {
        if (!status) return;

        const current = status.current_epoch || 0;
        const total = status.total_epochs || 1;
        const percent = Math.min(100, Math.round((current / total) * 100));

        if (trainingProgressBar) {
            trainingProgressBar.style.width = `${percent}%`;
            trainingProgressBar.setAttribute('aria-valuenow', percent);
        }

        if (trainingProgressText) {
            trainingProgressText.textContent = `${percent}% (Epoch ${current}/${total})`;
        }

        if (trainingStatusMessage) {
            trainingStatusMessage.textContent = status.status_message || 'Processing...';
        }

        // Update live metrics cards if present
        const trainAccBadge = document.getElementById('liveTrainAcc');
        const valAccBadge = document.getElementById('liveValAcc');
        const trainLossBadge = document.getElementById('liveTrainLoss');
        const valLossBadge = document.getElementById('liveValLoss');

        if (trainAccBadge) trainAccBadge.textContent = `${status.train_acc || 0.0}%`;
        if (valAccBadge) valAccBadge.textContent = `${status.val_acc || 0.0}%`;
        if (trainLossBadge) trainLossBadge.textContent = status.train_loss || '0.0000';
        if (valLossBadge) valLossBadge.textContent = status.val_loss || '0.0000';

        // Update epoch history rows
        if (epochHistoryTable && status.history && status.history.length > 0) {
            epochHistoryTable.innerHTML = '';
            status.history.forEach(item => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="fw-bold">Epoch ${item.epoch}</td>
                    <td>${item.train_loss}</td>
                    <td class="text-primary fw-semibold">${item.train_acc}%</td>
                    <td>${item.val_loss}</td>
                    <td class="text-success fw-bold">${item.val_acc}%</td>
                `;
                epochHistoryTable.appendChild(tr);
            });
        }
    }

    function resetTrainingButton() {
        if (startTrainingBtn) {
            startTrainingBtn.disabled = false;
            startTrainingBtn.innerHTML = `<i class="bi bi-play-circle-fill me-2"></i>Start Model Training`;
        }
    }
});
