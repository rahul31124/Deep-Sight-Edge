document.addEventListener('DOMContentLoaded', () => {
    // Telemetry Elements
    const liveFps = document.getElementById('live-fps');
    const liveEnh = document.getElementById('live-enh');
    const liveYolo = document.getElementById('live-yolo');
    const detectionTag = document.getElementById('detection-tag');

    // Census Counters
    const countTrash = document.getElementById('count-trash');
    const countBio = document.getElementById('count-bio');
    const countGear = document.getElementById('count-gear');

    // Pipeline Controls
    const toggleSbs = document.getElementById('toggle-sbs');
    const toggleEnh = document.getElementById('toggle-enh');
    const btnOrientH = document.getElementById('btn-orient-h');
    const btnOrientV = document.getElementById('btn-orient-v');
    const btnModelYolo8 = document.getElementById('btn-model-yolo8');
    const btnModelYolo26 = document.getElementById('btn-model-yolo26');
    const orientationRow = document.getElementById('orientation-row');
    const btnTheater = document.getElementById('btn-theater');
    const mainLayout = document.getElementById('main-layout');

    // Sliders
    const confSlider = document.getElementById('conf-slider');
    const confVal = document.getElementById('conf-val');
    const redSlider = document.getElementById('red-slider');
    const redVal = document.getElementById('red-val');
    const claheSlider = document.getElementById('clahe-slider');
    const claheVal = document.getElementById('clahe-val');

    // Video Source & Upload Elements
    const currentSourceLabel = document.getElementById('current-source-label');
    const videoSelect = document.getElementById('video-select');
    const btnLoadVideo = document.getElementById('btn-load-video');
    const btnSwitchCamera = document.getElementById('btn-switch-camera');
    const uploadDropZone = document.getElementById('upload-drop-zone');
    const videoFileInput = document.getElementById('video-file-input');
    const uploadStatus = document.getElementById('upload-status');

    // Raspberry Pi Helper Elements
    const piCameraInput = document.getElementById('pi-camera-input');
    const piModelInput = document.getElementById('pi-model-input');
    const piCmdDisplay = document.getElementById('pi-cmd-display');
    const btnCopyPiCmd = document.getElementById('btn-copy-pi-cmd');

    // --- Dynamic Telemetry Sync ---
    async function updateTelemetry() {
        try {
            const resp = await fetch('/api/stats');
            if (!resp.ok) return;
            const data = await resp.json();

            if (liveFps) liveFps.textContent = data.fps ? data.fps.toFixed(1) : '--';
            if (liveEnh) liveEnh.textContent = data.enhancement_ms ? data.enhancement_ms.toFixed(1) : '--';
            if (liveYolo) liveYolo.textContent = (data.inference_ms || data.yolo_ms) ? (data.inference_ms || data.yolo_ms).toFixed(1) : '--';
            if (detectionTag && data.model_name) detectionTag.textContent = data.model_name.toUpperCase() + ' INFERENCE';

            if (data.active_model) {
                if (data.active_model === 'yolo26n') {
                    if (btnModelYolo26) btnModelYolo26.classList.add('active');
                    if (btnModelYolo8) btnModelYolo8.classList.remove('active');
                } else {
                    if (btnModelYolo8) btnModelYolo8.classList.add('active');
                    if (btnModelYolo26) btnModelYolo26.classList.remove('active');
                }
            }

            if (currentSourceLabel && data.video_source) {
                currentSourceLabel.textContent = data.video_source;
            }

            if (data.counts) {
                if (countTrash) countTrash.textContent = data.counts["Trash"] || 0;
                if (countBio) countBio.textContent = data.counts["Bio"] || 0;
                if (countGear) countGear.textContent = data.counts["Submersible_Gear"] || 0;
            }
        } catch {
            // Silently retry on transient network interrupts
        }
    }

    setInterval(updateTelemetry, 400);
    updateTelemetry();

    // --- Config Sync for Sliders ---
    let syncTimeout = null;
    function sendConfigUpdate(params) {
        clearTimeout(syncTimeout);
        syncTimeout = setTimeout(async () => {
            try {
                await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params)
                });
            } catch (err) {
                console.error("Parameter update failed:", err);
            }
        }, 50);
    }

    // --- Instant Model Switching ---
    async function setModel(choice) {
        if (choice === 'yolo8n') {
            if (btnModelYolo8) btnModelYolo8.classList.add('active');
            if (btnModelYolo26) btnModelYolo26.classList.remove('active');
        } else {
            if (btnModelYolo26) btnModelYolo26.classList.add('active');
            if (btnModelYolo8) btnModelYolo8.classList.remove('active');
        }

        try {
            const resp = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_choice: choice })
            });
            const data = await resp.json();
            if (detectionTag && data.updated && data.updated.model_name) {
                detectionTag.textContent = data.updated.model_name.toUpperCase() + ' INFERENCE';
            }
        } catch (err) {
            console.error("Model switch failed:", err);
        }
    }

    if (btnModelYolo8 && btnModelYolo26) {
        btnModelYolo8.addEventListener('click', () => setModel('yolo8n'));
        btnModelYolo26.addEventListener('click', () => setModel('yolo26n'));
    }

    async function loadVideoList() {
        if (!videoSelect) return;
        try {
            const resp = await fetch('/api/videos');
            if (!resp.ok) return;
            const data = await resp.json();

            videoSelect.innerHTML = '';
            if (data.videos && data.videos.length > 0) {
                data.videos.forEach(vid => {
                    const opt = document.createElement('option');
                    opt.value = `videos/${vid}`;
                    opt.textContent = vid;
                    if (data.current && data.current === vid) {
                        opt.selected = true;
                    }
                    videoSelect.appendChild(opt);
                });
            } else {
                const opt = document.createElement('option');
                opt.value = '';
                opt.textContent = 'No videos found in videos/';
                videoSelect.appendChild(opt);
            }
        } catch (err) {
            console.error("Failed to fetch video list:", err);
        }
    }

    loadVideoList();

    async function changeVideoSource(source) {
        try {
            const resp = await fetch('/api/change_source', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source: source })
            });
            const data = await resp.json();
            if (data.status === 'success') {
                if (currentSourceLabel) currentSourceLabel.textContent = source;
            }
        } catch (err) {
            console.error("Failed to switch video source:", err);
        }
    }

    if (btnLoadVideo && videoSelect) {
        btnLoadVideo.addEventListener('click', () => {
            if (videoSelect.value) {
                changeVideoSource(videoSelect.value);
            }
        });
        videoSelect.addEventListener('change', () => {
            if (videoSelect.value) {
                changeVideoSource(videoSelect.value);
            }
        });
    }

    if (btnSwitchCamera) {
        btnSwitchCamera.addEventListener('click', () => {
            changeVideoSource('/dev/video0');
        });
    }

    if (uploadDropZone && videoFileInput) {
        uploadDropZone.addEventListener('click', () => {
            videoFileInput.click();
        });

        uploadDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadDropZone.classList.add('drag-over');
        });

        uploadDropZone.addEventListener('dragleave', () => {
            uploadDropZone.classList.remove('drag-over');
        });

        uploadDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadDropZone.classList.remove('drag-over');
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });

        videoFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });
    }

    async function handleFileUpload(file) {
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);

        if (uploadStatus) {
            uploadStatus.textContent = `Uploading ${file.name}...`;
            uploadStatus.className = 'upload-status-text';
        }

        try {
            const resp = await fetch('/api/upload_video', {
                method: 'POST',
                body: formData
            });
            const result = await resp.json();
            if (resp.ok && result.status === 'success') {
                if (uploadStatus) {
                    uploadStatus.textContent = `Uploaded ${result.filename}! Feed active.`;
                    uploadStatus.className = 'upload-status-text success';
                }
                if (videoFileInput) videoFileInput.value = '';
                loadVideoList();
            } else {
                if (uploadStatus) {
                    uploadStatus.textContent = result.message || 'Upload failed.';
                    uploadStatus.className = 'upload-status-text error';
                }
            }
        } catch (err) {
            if (uploadStatus) {
                uploadStatus.textContent = 'Upload network error: ' + err.message;
                uploadStatus.className = 'upload-status-text error';
            }
        }
    }
    if (toggleSbs) {
        toggleSbs.addEventListener('change', (e) => {
            const checked = e.target.checked;
            if (orientationRow) {
                orientationRow.style.opacity = checked ? '1' : '0.4';
                orientationRow.style.pointerEvents = checked ? 'auto' : 'none';
            }
            sendConfigUpdate({ side_by_side: checked });
        });
    }

    if (btnOrientH && btnOrientV) {
        btnOrientH.addEventListener('click', () => {
            btnOrientH.classList.add('active');
            btnOrientV.classList.remove('active');
            sendConfigUpdate({ split_orientation: 'horizontal' });
        });

        btnOrientV.addEventListener('click', () => {
            btnOrientV.classList.add('active');
            btnOrientH.classList.remove('active');
            sendConfigUpdate({ split_orientation: 'vertical' });
        });
    }

    if (btnTheater && mainLayout) {
        btnTheater.addEventListener('click', () => {
            mainLayout.classList.toggle('theater-mode');
            const isTheater = mainLayout.classList.contains('theater-mode');
            btnTheater.classList.toggle('active', isTheater);
            const textSpan = btnTheater.querySelector('.theater-text');
            if (textSpan) {
                textSpan.textContent = isTheater ? 'Compact View' : 'Expand Canvas';
            }
        });
    }

    if (toggleEnh) {
        toggleEnh.addEventListener('change', (e) => {
            sendConfigUpdate({ apply_enhancement: e.target.checked });
        });
    }

    if (confSlider && confVal) {
        confSlider.addEventListener('input', (e) => {
            confVal.textContent = parseFloat(e.target.value).toFixed(2);
            sendConfigUpdate({ conf_thresh: parseFloat(e.target.value) });
        });
    }

    if (redSlider && redVal) {
        redSlider.addEventListener('input', (e) => {
            redVal.textContent = `${parseFloat(e.target.value).toFixed(2)}x`;
            sendConfigUpdate({ red_gain: parseFloat(e.target.value) });
        });
    }

    if (claheSlider && claheVal) {
        claheSlider.addEventListener('input', (e) => {
            claheVal.textContent = parseFloat(e.target.value).toFixed(1);
            sendConfigUpdate({ clip_limit: parseFloat(e.target.value) });
        });
    }
    function updatePiCommand() {
        if (!piCmdDisplay) return;
        const cam = piCameraInput ? piCameraInput.value : '/dev/video0';
        const model = piModelInput ? piModelInput.value : 'yolo26n.pt';
        piCmdDisplay.textContent = `python3 main.py --video ${cam} --model ${model} --port 8000`;
    }

    if (piCameraInput) piCameraInput.addEventListener('change', updatePiCommand);
    if (piModelInput) piModelInput.addEventListener('change', updatePiCommand);

    if (btnCopyPiCmd && piCmdDisplay) {
        btnCopyPiCmd.addEventListener('click', async () => {
            const textToCopy = piCmdDisplay.textContent;
            try {
                await navigator.clipboard.writeText(textToCopy);
                const originalText = btnCopyPiCmd.textContent;
                btnCopyPiCmd.textContent = 'Copied!';
                btnCopyPiCmd.classList.add('copied');
                setTimeout(() => {
                    btnCopyPiCmd.textContent = originalText;
                    btnCopyPiCmd.classList.remove('copied');
                }, 2000);
            } catch (err) {
                const textarea = document.createElement('textarea');
                textarea.value = textToCopy;
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand('copy');
                document.body.removeChild(textarea);
                btnCopyPiCmd.textContent = 'Copied!';
                setTimeout(() => { btnCopyPiCmd.textContent = 'Copy'; }, 2000);
            }
        });
    }
});
