const classNames = [
    'USD_1', 'USD_5', 'USD_10', 'USD_20', 'USD_50', 'USD_100', 
    'KHR_100', 'KHR_500', 'KHR_1000', 'KHR_2000', 'KHR_5000', 'KHR_10000', 'KHR_20000', 'KHR_50000'
];

const CLASS_VALUES = {
    USD_1: { currency: 'USD', value: 1 },
    USD_5: { currency: 'USD', value: 5 },
    USD_10: { currency: 'USD', value: 10 },
    USD_20: { currency: 'USD', value: 20 },
    USD_50: { currency: 'USD', value: 50 },
    USD_100: { currency: 'USD', value: 100 },
    KHR_100: { currency: 'KHR', value: 100 },
    KHR_500: { currency: 'KHR', value: 500 },
    KHR_1000: { currency: 'KHR', value: 1000 },
    KHR_2000: { currency: 'KHR', value: 2000 },
    KHR_5000: { currency: 'KHR', value: 5000 },
    KHR_10000: { currency: 'KHR', value: 10000 },
    KHR_20000: { currency: 'KHR', value: 20000 },
    KHR_50000: { currency: 'KHR', value: 50000 }
};

let session = null;
let isModelLoading = false;
let CONF_THRESHOLD = 0.20;
let currentBoxes = [];
let currentInferenceTime = 0;
const MODEL_URL = '../models/rielvision.onnx';

const uploadInput = document.getElementById('demo-upload');
const uploadState = document.getElementById('upload-state');
const demoImage = document.getElementById('demo-image');
const demoCanvas = document.getElementById('demo-canvas');
const statusText = document.getElementById('tv-status');
const valueText = document.getElementById('tv-value');
const resetBtn = document.getElementById('reset-btn');
const demoContainer = document.getElementById('demo-container');

const confSlider = document.getElementById('conf-slider');
const confVal = document.getElementById('conf-val');

// YOLO26n browser input parameters
const MODEL_DIM = 640;

// Initialize slider value
if (confSlider) {
    confSlider.value = CONF_THRESHOLD;
    confVal.innerText = CONF_THRESHOLD.toFixed(2);
    confSlider.addEventListener('input', (e) => {
        CONF_THRESHOLD = parseFloat(e.target.value);
        confVal.innerText = CONF_THRESHOLD.toFixed(2);
        if (currentBoxes.length > 0) {
            drawBoxes(currentBoxes, currentInferenceTime);
        }
    });
}

async function initModel() {
    if (session) return;
    isModelLoading = true;
    
    const progressContainer = document.getElementById('model-progress-container');
    const progressBar = document.getElementById('model-progress-bar');
    const progressText = document.getElementById('model-progress-text');
    
    if (progressContainer) progressContainer.classList.remove('hidden');
    statusText.innerText = 'DOWNLOADING MODEL...';
    statusText.className = 'text-k-red font-bold uppercase animate-pulse';
    
    try {
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        
        // Fetch model with progress tracking
        const response = await fetch(MODEL_URL);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const contentLength = response.headers.get('content-length');
        const total = contentLength ? parseInt(contentLength, 10) : 0;
        
        const reader = response.body.getReader();
        let loaded = 0;
        const chunks = [];
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            chunks.push(value);
            loaded += value.length;
            
            if (total > 0) {
                const percent = (loaded / total) * 100;
                if (progressBar) progressBar.style.width = `${percent}%`;
                if (progressText) progressText.innerText = `DOWNLOADING: ${percent.toFixed(0)}%`;
            } else {
                const loadedMB = (loaded / (1024 * 1024)).toFixed(1);
                if (progressText) progressText.innerText = `DOWNLOADING: ${loadedMB}MB`;
            }
        }
        
        if (progressText) progressText.innerText = 'INITIALIZING...';
        statusText.innerText = 'INITIALIZING...';
        
        // Combine chunks into a single Uint8Array
        const modelData = new Uint8Array(loaded);
        let position = 0;
        for (const chunk of chunks) {
            modelData.set(chunk, position);
            position += chunk.length;
        }
        
        session = await ort.InferenceSession.create(modelData.buffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        
        console.log("Model loaded successfully.");
        statusText.innerText = 'READY';
        statusText.className = 'text-green-600 font-bold uppercase';
        
        // Hide the progress bar container when done
        if (progressContainer) {
            setTimeout(() => {
                progressContainer.style.opacity = '0';
                progressContainer.style.transition = 'opacity 0.5s ease';
                setTimeout(() => {
                    progressContainer.classList.add('hidden');
                    progressContainer.style.opacity = '1'; // Reset for next load if any
                }, 500);
            }, 1000);
        }
    } catch (e) {
        console.error("Failed to load model:", e);
        statusText.innerText = 'ERROR LOADING MODEL';
        statusText.className = 'text-k-red font-bold uppercase';
        if (progressText) progressText.innerText = 'FAILED';
    }
    isModelLoading = false;
}

async function handleImageFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please upload a valid image file.');
        return;
    }

    const objectUrl = URL.createObjectURL(file);
    demoImage.src = objectUrl;
    demoImage.classList.remove('hidden');
    uploadState.style.display = 'none';
    resetBtn.classList.remove('hidden');
    
    // Clear previous boxes immediately
    const ctx = demoCanvas.getContext('2d');
    ctx.clearRect(0, 0, demoCanvas.width, demoCanvas.height);
    
    demoImage.onload = async () => {
        if (!session && !isModelLoading) {
            initModel();
        }
        
        while(!session) {
            if (!isModelLoading) return;
            await new Promise(r => setTimeout(r, 100));
        }
        
        await runInference();
    };
}

uploadInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        await handleImageFile(file);
    }
});

resetBtn.addEventListener('click', () => {
    // Instead of resetting the UI to empty, just trigger the file selector
    uploadInput.click();
});

// Drag and drop event listeners
if (demoContainer) {
    ['dragenter', 'dragover'].forEach(eventName => {
        demoContainer.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            demoContainer.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        demoContainer.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            demoContainer.classList.remove('drag-over');
        }, false);
    });

    demoContainer.addEventListener('drop', async (e) => {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        if (file) {
            await handleImageFile(file);
        }
    }, false);
}

async function runInference() {
    if (!session) return;
    
    statusText.innerText = 'ANALYZING...';
    statusText.className = 'text-k-red font-bold uppercase animate-pulse';
    if (valueText) valueText.innerText = '...';
    
    const [tensor, scale, padX, padY] = preprocessImage(demoImage, MODEL_DIM);
    
    const start = performance.now();
    const feeds = { images: tensor };
    const results = await session.run(feeds);
    
    const outputName = session.outputNames[0];
    const outputTensor = results[outputName];
    currentInferenceTime = performance.now() - start;
    
    currentBoxes = parseOutput(outputTensor, scale, padX, padY, demoImage.naturalWidth, demoImage.naturalHeight);
    
    drawBoxes(currentBoxes, currentInferenceTime);
}

function preprocessImage(img, targetDim) {
    const canvas = document.createElement('canvas');
    canvas.width = targetDim;
    canvas.height = targetDim;
    const ctx = canvas.getContext('2d');
    
    const scale = Math.min(targetDim / img.naturalWidth, targetDim / img.naturalHeight);
    const w = img.naturalWidth * scale;
    const h = img.naturalHeight * scale;
    const padX = (targetDim - w) / 2;
    const padY = (targetDim - h) / 2;
    
    ctx.fillStyle = '#727272';
    ctx.fillRect(0, 0, targetDim, targetDim);
    ctx.drawImage(img, padX, padY, w, h);
    
    const imgData = ctx.getImageData(0, 0, targetDim, targetDim).data;
    const float32Data = new Float32Array(3 * targetDim * targetDim);
    for (let i = 0; i < targetDim * targetDim; i++) {
        float32Data[i] = imgData[i * 4] / 255.0; // R
        float32Data[targetDim * targetDim + i] = imgData[i * 4 + 1] / 255.0; // G
        float32Data[2 * targetDim * targetDim + i] = imgData[i * 4 + 2] / 255.0; // B
    }
    
    const tensor = new ort.Tensor('float32', float32Data, [1, 3, targetDim, targetDim]);
    return [tensor, scale, padX, padY];
}

function parseOutput(outputTensor, scale, padX, padY, origW, origH) {
    const outputData = outputTensor.data;
    const dims = outputTensor.dims;
    const boxes = [];
    
    // Check if the model has NMS built-in [1, N, 6] or if it is raw logits [1, 17, 8400]
    if (dims[2] === 6 || dims.length === 2 && dims[1] === 6) {
        // Handle NMS-enabled model format [1, N, 6] or [N, 6]
        const numBoxes = dims.length === 3 ? dims[1] : dims[0];
        for (let i = 0; i < numBoxes; i++) {
            const offset = i * 6;
            const score = outputData[offset + 4];
            
            const classId = Math.round(outputData[offset + 5]);
            
            const x1 = (outputData[offset] - padX) / scale;
            const y1 = (outputData[offset + 1] - padY) / scale;
            const x2 = (outputData[offset + 2] - padX) / scale;
            const y2 = (outputData[offset + 3] - padY) / scale;
            
            if (score >= 0.01) {
                boxes.push({
                    x1: Math.max(0, x1),
                    y1: Math.max(0, y1),
                    x2: Math.min(origW, x2),
                    y2: Math.min(origH, y2),
                    score: score,
                    classId: classId
                });
            }
        }
    } else {
        // Fallback for raw model [1, 17, 8400]
        let isTransposed = false;
        let numBoxes = 8400;
        let numElements = 17;
        
        if (dims[1] === 8400 || dims[1] === 8400) {
            isTransposed = true;
            numBoxes = dims[1];
            numElements = dims[2];
        } else {
            numBoxes = dims[2];
            numElements = dims[1];
        }

        const numClasses = classNames.length;
        
        for (let i = 0; i < numBoxes; i++) {
            let maxClassProb = 0;
            let classId = -1;
            
            for (let c = 0; c < numClasses; c++) {
                const idx = isTransposed ? (i * numElements + 4 + c) : ((4 + c) * numBoxes + i);
                const prob = outputData[idx];
                if (prob > maxClassProb) {
                    maxClassProb = prob;
                    classId = c;
                }
            }
            
            if (maxClassProb >= 0.01) {
                const cxIdx = isTransposed ? (i * numElements + 0) : (0 * numBoxes + i);
                const cyIdx = isTransposed ? (i * numElements + 1) : (1 * numBoxes + i);
                const wIdx  = isTransposed ? (i * numElements + 2) : (2 * numBoxes + i);
                const hIdx  = isTransposed ? (i * numElements + 3) : (3 * numBoxes + i);

                const cx = outputData[cxIdx];
                const cy = outputData[cyIdx];
                const w = outputData[wIdx];
                const h = outputData[hIdx];
                
                const x1 = ((cx - w/2) - padX) / scale;
                const y1 = ((cy - h/2) - padY) / scale;
                const x2 = ((cx + w/2) - padX) / scale;
                const y2 = ((cy + h/2) - padY) / scale;
                
                boxes.push({
                    x1: Math.max(0, x1),
                    y1: Math.max(0, y1),
                    x2: Math.min(origW, x2),
                    y2: Math.min(origH, y2),
                    score: maxClassProb,
                    classId: classId
                });
            }
        }
        
        return applyNMS(boxes);
    }
    
    return applyNMS(boxes);
}

function applyNMS(boxes) {
    // Class-agnostic Non-Maximum Suppression (NMS)
    // Sort all boxes by confidence score descending, regardless of class
    let sortedBoxes = [...boxes].sort((a, b) => b.score - a.score);
    const finalBoxes = [];
    
    while (sortedBoxes.length > 0) {
        const bestBox = sortedBoxes[0];
        finalBoxes.push(bestBox);
        sortedBoxes.splice(0, 1);
        
        sortedBoxes = sortedBoxes.filter(box => {
            const iou = calculateIOU(bestBox, box);
            return iou < 0.45; // Reject overlapping boxes with IoU >= 0.45
        });
    }
    return finalBoxes;
}

function calculateIOU(box1, box2) {
    const interX1 = Math.max(box1.x1, box2.x1);
    const interY1 = Math.max(box1.y1, box2.y1);
    const interX2 = Math.min(box1.x2, box2.x2);
    const interY2 = Math.min(box1.y2, box2.y2);
    
    const interW = Math.max(0, interX2 - interX1);
    const interH = Math.max(0, interY2 - interY1);
    const interArea = interW * interH;
    
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    
    return interArea / (area1 + area2 - interArea);
}

function formatTotals(boxes) {
    const totals = { KHR: 0, USD: 0 };
    for (const box of boxes) {
        const className = classNames[box.classId];
        const info = CLASS_VALUES[className];
        if (!info) continue;
        totals[info.currency] += info.value;
    }

    const khr = totals.KHR.toLocaleString('en-US');
    const usd = totals.USD.toLocaleString('en-US');
    return `KHR ${khr} / USD ${usd}`;
}

function drawBoxes(allBoxes, inferenceTime) {
    demoCanvas.width = demoImage.naturalWidth;
    demoCanvas.height = demoImage.naturalHeight;
    demoCanvas.classList.remove('hidden');
    
    const ctx = demoCanvas.getContext('2d');
    ctx.clearRect(0, 0, demoCanvas.width, demoCanvas.height);
    
    const filteredBoxes = allBoxes.filter(box => box.score >= CONF_THRESHOLD);
    
    // Keep the visible box/tag size stable across tiny screenshots and huge photos.
    const imageRect = demoImage.getBoundingClientRect();
    const displayScale = Math.max(
        0.01,
        Math.min(
            imageRect.width / demoImage.naturalWidth,
            imageRect.height / demoImage.naturalHeight
        )
    );
    const lineWidth = Math.max(1, Math.round(2 / displayScale));
    const fontSize = Math.max(10, Math.round(12 / displayScale));
    ctx.lineWidth = lineWidth;
    ctx.font = `bold ${fontSize}px "JetBrains Mono", "Space Mono", monospace`;
    ctx.textBaseline = 'top';
    
    const COLORS = [
        "#e43d30", "#2478c2", "#f0a202", "#1b998b", "#9b5de5", 
        "#ef476f", "#06d6a0", "#ffd166", "#118ab2", "#f78c6b", 
        "#00b4d8", "#f72585", "#90be6d", "#4361ee"
    ];
    
    // Draw every box first, then labels last so label tags sit above all lines.
    filteredBoxes.forEach(box => {
        const color = COLORS[box.classId % COLORS.length];
        
        ctx.strokeStyle = color;
        
        const w = box.x2 - box.x1;
        const h = box.y2 - box.y1;
        ctx.strokeRect(box.x1, box.y1, w, h);
    });

    const drawnLabels = [];
    
    filteredBoxes.forEach(box => {
        const className = classNames[box.classId] || `CLASS_${box.classId}`;
        const color = COLORS[box.classId % COLORS.length];
        const label = `${className} (${(box.score*100).toFixed(0)}%)`;
        const textWidth = ctx.measureText(label).width;
        const labelW = textWidth + 8;
        const labelH = fontSize + 5;
        
        // Smart label placement (try multiple positions to avoid overlap)
        const positions = [
            { x: box.x1, y: box.y1 - labelH },                // Top-Left (Outside)
            { x: box.x1, y: box.y1 + lineWidth },             // Top-Left (Inside)
            { x: box.x2 - labelW, y: box.y1 - labelH },       // Top-Right (Outside)
            { x: box.x2 - labelW, y: box.y1 + lineWidth },    // Top-Right (Inside)
            { x: box.x1, y: box.y2 - labelH - lineWidth },    // Bottom-Left (Inside)
            { x: box.x2 - labelW, y: box.y2 - labelH - lineWidth } // Bottom-Right (Inside)
        ];
        
        let finalPos = {
            x: Math.max(0, Math.min(positions[0].x, demoCanvas.width - labelW)),
            y: Math.max(0, Math.min(positions[0].y, demoCanvas.height - labelH))
        };
        for (let pos of positions) {
            // Screen bounds check
            if (pos.x < 0 || pos.y < 0 || pos.x + labelW > demoCanvas.width || pos.y + labelH > demoCanvas.height) {
                continue;
            }
            // Collision check
            let collision = false;
            for (let drawn of drawnLabels) {
                if (!(pos.x + labelW < drawn.x || pos.x > drawn.x + drawn.w || 
                      pos.y + labelH < drawn.y || pos.y > drawn.y + drawn.h)) {
                    collision = true;
                    break;
                }
            }
            if (!collision) {
                finalPos = pos;
                break;
            }
        }
        
        // If all overlap, just use the first one but it will overlap
        drawnLabels.push({ x: finalPos.x, y: finalPos.y, w: labelW, h: labelH });
        
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = color;
        ctx.fillRect(finalPos.x, finalPos.y, labelW, labelH);
        
        // Use white or dark text depending on color contrast, but white usually looks good
        ctx.fillStyle = (color === "#ffd166" || color === "#06d6a0") ? '#000000' : '#FFFFFF';
        ctx.fillText(label, finalPos.x + 4, finalPos.y + 2);
    });
    
    statusText.innerText = `COMPLETED IN ${(inferenceTime / 1000).toFixed(2)}S`;
    statusText.className = 'text-green-600 font-bold uppercase';
    
    if (valueText) {
        valueText.innerText = formatTotals(filteredBoxes);
        valueText.className = filteredBoxes.length > 0 ? 'font-bold text-k-red uppercase' : 'font-bold text-k-blue uppercase';
    }
}

// Start loading the model in the background immediately on page load
initModel();

