document.addEventListener('DOMContentLoaded', () => {
    const liveFps = document.getElementById('live-fps');
    const liveEnh = document.getElementById('live-enh');
    const liveYolo = document.getElementById('live-yolo');

    const countTrash = document.getElementById('count-trash');
    const countBio = document.getElementById('count-bio');
    const countGear = document.getElementById('count-gear');

    const toggleSbs = document.getElementById('toggle-sbs');
    const toggleEnh = document.getElementById('toggle-enh');
    const btnOrientH = document.getElementById('btn-orient-h');
    const btnOrientV = document.getElementById('btn-orient-v');
    const orientationRow = document.getElementById('orientation-row');
    const btnTheater = document.getElementById('btn-theater');
    const mainLayout = document.getElementById('main-layout');

    const confSlider = document.getElementById('conf-slider');
    const confVal = document.getElementById('conf-val');
    const redSlider = document.getElementById('red-slider');
    const redVal = document.getElementById('red-val');
    const claheSlider = document.getElementById('clahe-slider');
    const claheVal = document.getElementById('clahe-val');

    async function updateTelemetry() {
        try {
            const resp = await fetch('/api/stats');
            if (!resp.ok) return;
            const data = await resp.json();

            if (liveFps) liveFps.textContent = data.fps ? data.fps.toFixed(1) : '--';
            if (liveEnh) liveEnh.textContent = data.enhancement_ms ? data.enhancement_ms.toFixed(1) : '--';
            if (liveYolo) liveYolo.textContent = data.yolo_ms ? data.yolo_ms.toFixed(1) : '--';

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
                textSpan.textContent = isTheater ? 'Standard Canvas' : 'Expand Canvas';
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
});
