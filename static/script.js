/**
 * Brain Tumor Segmentation - Frontend JavaScript
 * Handles file upload, image preview, API calls, and result display
 */

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
    // Upload section
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    browseBtn: document.getElementById('browseBtn'),
    previewArea: document.getElementById('previewArea'),
    previewImage: document.getElementById('previewImage'),
    fileName: document.getElementById('fileName'),
    changeBtn: document.getElementById('changeBtn'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    
    // Results section
    resultsSection: document.getElementById('resultsSection'),
    statusBanner: document.getElementById('statusBanner'),
    statusIcon: document.getElementById('statusIcon'),
    statusTitle: document.getElementById('statusTitle'),
    statusDescription: document.getElementById('statusDescription'),
    metricsGrid: document.getElementById('metricsGrid'),
    confidenceValue: document.getElementById('confidenceValue'),
    confidenceBar: document.getElementById('confidenceBar'),
    tumorRatioValue: document.getElementById('tumorRatioValue'),
    tumorRatioBar: document.getElementById('tumorRatioBar'),
    imagesDisplay: document.getElementById('imagesDisplay'),
    resultOriginal: document.getElementById('resultOriginal'),
    resultMask: document.getElementById('resultMask'),
    resultOverlay: document.getElementById('resultOverlay'),
    newAnalysisBtn: document.getElementById('newAnalysisBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    
    // Error section
    errorSection: document.getElementById('errorSection'),
    errorMessage: document.getElementById('errorMessage'),
    retryBtn: document.getElementById('retryBtn')
};

// ============================================================================
// State
// ============================================================================

let currentFile = null;
let lastResult = null;

// ============================================================================
// Event Listeners
// ============================================================================

// File input change
elements.fileInput.addEventListener('change', handleFileSelect);

// Browse button click
elements.browseBtn.addEventListener('click', () => elements.fileInput.click());

// Upload area click
elements.uploadArea.addEventListener('click', (e) => {
    if (e.target !== elements.browseBtn) {
        elements.fileInput.click();
    }
});

// Drag and drop
elements.uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.uploadArea.classList.add('dragover');
});

elements.uploadArea.addEventListener('dragleave', () => {
    elements.uploadArea.classList.remove('dragover');
});

elements.uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Change button
elements.changeBtn.addEventListener('click', resetUpload);

// Analyze button
elements.analyzeBtn.addEventListener('click', analyzeImage);

// New analysis button
elements.newAnalysisBtn.addEventListener('click', resetAll);

// Download button
elements.downloadBtn.addEventListener('click', downloadResults);

// Retry button
elements.retryBtn.addEventListener('click', resetAll);

// ============================================================================
// File Handling
// ============================================================================

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff', 'image/gif'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload a PNG, JPG, BMP, or TIFF image.');
        return;
    }
    
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File too large. Maximum size is 16MB.');
        return;
    }
    
    currentFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.previewImage.src = e.target.result;
        elements.fileName.textContent = file.name;
        
        elements.uploadArea.classList.add('hidden');
        elements.previewArea.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    currentFile = null;
    elements.fileInput.value = '';
    elements.previewImage.src = '';
    
    elements.previewArea.classList.add('hidden');
    elements.uploadArea.classList.remove('hidden');
}

function resetAll() {
    resetUpload();
    lastResult = null;
    
    elements.resultsSection.classList.add('hidden');
    elements.errorSection.classList.add('hidden');
}

// ============================================================================
// API Calls
// ============================================================================

async function analyzeImage() {
    if (!currentFile) {
        showError('Please select an image first.');
        return;
    }
    
    // Show loading state
    setButtonLoading(elements.analyzeBtn, true);
    elements.errorSection.classList.add('hidden');
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', currentFile);
        
        // Send request
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Analysis failed');
        }
        
        // Store result and display
        lastResult = data;
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred during analysis.');
    } finally {
        setButtonLoading(elements.analyzeBtn, false);
    }
}

// ============================================================================
// Display Results
// ============================================================================

function displayResults(data) {
    // Hide upload section, show results
    elements.uploadArea.classList.add('hidden');
    elements.previewArea.classList.add('hidden');
    elements.errorSection.classList.add('hidden');
    elements.resultsSection.classList.remove('hidden');
    
    // Update status banner
    updateStatusBanner(data.tumor_detected, data.confidence_percent);
    
    // Update metrics
    updateMetrics(data);
    
    // Update images
    elements.resultOriginal.src = data.original_image;
    elements.resultMask.src = data.mask_image;
    elements.resultOverlay.src = data.overlay_image;
}

function updateStatusBanner(tumorDetected, confidence) {
    // Remove existing classes
    elements.statusBanner.classList.remove('success', 'danger', 'warning');
    
    if (tumorDetected) {
        if (confidence >= 70) {
            elements.statusBanner.classList.add('danger');
            elements.statusTitle.textContent = 'Tumor Detected';
            elements.statusDescription.textContent = 
                `High confidence detection (${confidence.toFixed(1)}%). Please consult a medical professional.`;
            elements.statusIcon.innerHTML = getIconSVG('alert');
        } else {
            elements.statusBanner.classList.add('warning');
            elements.statusTitle.textContent = 'Possible Tumor Detected';
            elements.statusDescription.textContent = 
                `Moderate confidence (${confidence.toFixed(1)}%). Further examination recommended.`;
            elements.statusIcon.innerHTML = getIconSVG('warning');
        }
    } else {
        elements.statusBanner.classList.add('success');
        elements.statusTitle.textContent = 'No Tumor Detected';
        elements.statusDescription.textContent = 
            'No significant tumor regions were identified in this scan.';
        elements.statusIcon.innerHTML = getIconSVG('check');
    }
}

function updateMetrics(data) {
    // Confidence score
    const confidence = data.confidence_percent || 0;
    elements.confidenceValue.textContent = `${confidence.toFixed(1)}%`;
    animateBar(elements.confidenceBar, Math.min(confidence, 100));
    
    // Tumor ratio
    const tumorRatio = data.tumor_ratio_percent || 0;
    elements.tumorRatioValue.textContent = `${tumorRatio.toFixed(2)}%`;
    animateBar(elements.tumorRatioBar, Math.min(tumorRatio * 10, 100)); // Scale for visibility
}

function animateBar(barElement, percentage) {
    setTimeout(() => {
        barElement.style.width = `${percentage}%`;
    }, 100);
}

// ============================================================================
// Error Handling
// ============================================================================

function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorSection.classList.remove('hidden');
    elements.resultsSection.classList.add('hidden');
}

// ============================================================================
// Download Results
// ============================================================================

function downloadResults() {
    if (!lastResult) return;
    
    // Create a link element to download the overlay image
    const link = document.createElement('a');
    link.href = lastResult.overlay_image;
    link.download = `brain_tumor_analysis_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// ============================================================================
// Utilities
// ============================================================================

function setButtonLoading(button, loading) {
    if (loading) {
        button.classList.add('loading');
        button.disabled = true;
    } else {
        button.classList.remove('loading');
        button.disabled = false;
    }
}

function getIconSVG(type) {
    const icons = {
        check: `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <path d="M9 12l2 2 4-4"/>
            </svg>
        `,
        alert: `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <path d="M12 8v4"/>
                <path d="M12 16h.01"/>
            </svg>
        `,
        warning: `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <path d="M12 9v4"/>
                <path d="M12 17h.01"/>
            </svg>
        `
    };
    return icons[type] || icons.alert;
}

// ============================================================================
// Initialize
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('Brain Tumor Segmentation UI loaded');
    
    // Check if API is healthy
    fetch('/health')
        .then(res => res.json())
        .then(data => {
            console.log('API Status:', data);
        })
        .catch(err => {
            console.warn('API health check failed:', err);
        });
});
