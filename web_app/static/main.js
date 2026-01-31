document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeImg = document.getElementById('removeImg');
    const analyzeBtn = document.getElementById('analyzeBtn');

    const resultsPlaceholder = document.getElementById('resultsPlaceholder');
    const loadingState = document.getElementById('loadingState');
    const resultsContent = document.getElementById('resultsContent');
    const probList = document.getElementById('probList');
    const gradcamImg = document.getElementById('gradcamImg');

    let selectedFile = null;

    // Handle Drag and Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-active');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-active');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-active');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    // Handle Browse
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    // Remove Image
    removeImg.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        resetResults();
    });

    function resetResults() {
        resultsContent.classList.add('hidden');
        resultsPlaceholder.classList.remove('hidden');
        probList.innerHTML = '';
        gradcamImg.src = '';
    }

    // Analyze
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        resetResults();
        resultsPlaceholder.classList.add('hidden');
        loadingState.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            console.log('Sending request to: /api/predict');
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error(`Server Error: ${response.status}`, errorText);
                throw new Error(`Server returned ${response.status}: ${errorText}`);
            }

            const data = await response.json();
            console.log('Analysis result received:', data);
            displayResults(data);
        } catch (error) {
            console.error('Detailed Analysis Error:', error);
            if (error instanceof TypeError && (error.message.includes('fetch') || error.message.includes('NetworkError'))) {
                alert(`Connectivity Error: The frontend could not reach the backend.\n\n` +
                    `1. Ensure the Python server is running.\n` +
                    `2. Ensure you are at http://localhost:8010 (Current: ${window.location.origin})\n` +
                    `3. Check the browser console (F12) for details.`);
            } else {
                alert(`Analysis failed: ${error.message}`);
            }
            resetResults();
        } finally {
            loadingState.classList.add('hidden');
        }
    });

    // Fetch Model Accuracy
    async function fetchModelAccuracy() {
        try {
            const response = await fetch('/health');
            if (response.ok) {
                const data = await response.json();
                if (data.model_accuracy) {
                    const accuracyPercentage = (data.model_accuracy * 100).toFixed(1);
                    const accEl = document.getElementById('modelAccuracy');
                    const accBadge = document.getElementById('accuracyBadge');
                    if (accEl && accBadge) {
                        accEl.textContent = `${accuracyPercentage}%`;
                        accBadge.classList.remove('hidden');
                    }
                }
            }
        } catch (error) {
            console.error('Failed to fetch model accuracy:', error);
        }
    }

    // Call fetchModelAccuracy on load
    fetchModelAccuracy();

    function displayResults(data) {
        probList.innerHTML = '';

        // Display Confidence of top prediction
        if (data.predictions && data.predictions.length > 0) {
            const topPred = data.predictions[0];
            const confidencePercentage = (topPred.probability * 100).toFixed(1);
            const confEl = document.getElementById('predConfidence');
            const confBadge = document.getElementById('confidenceBadge');
            if (confEl && confBadge) {
                confEl.textContent = `${confidencePercentage}%`;
                confBadge.classList.remove('hidden');
            }
        }

        data.predictions.forEach(pred => {
            const percentage = (pred.probability * 100).toFixed(1);
            const item = document.createElement('div');
            item.className = 'prob-item';
            item.innerHTML = `
                <div class="prob-info">
                    <span class="prob-label">${pred.label}</span>
                    <span class="prob-value">${percentage}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: 0%"></div>
                </div>
            `;
            probList.appendChild(item);

            // Animate progress bar
            setTimeout(() => {
                item.querySelector('.prob-bar-fill').style.width = `${percentage}%`;
            }, 100);
        });

        gradcamImg.src = data.heatmap_url + '?t=' + Date.now(); // Cache busting
        resultsContent.classList.remove('hidden');
    }
});
