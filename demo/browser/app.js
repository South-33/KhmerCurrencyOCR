const STACK_CONFIG_URL = "/configs/cashsnap_two_stage_oldcommon_browser_stack.json";
const DEFAULT_CLASS_NAMES = [
  "USD_1",
  "USD_5",
  "USD_10",
  "USD_20",
  "USD_50",
  "USD_100",
  "KHR_500",
  "KHR_1000",
  "KHR_2000",
  "KHR_5000",
  "KHR_10000",
  "KHR_20000",
  "KHR_50000",
];
const VALUES = {
  USD_1: 1,
  USD_5: 5,
  USD_10: 10,
  USD_20: 20,
  USD_50: 50,
  USD_100: 100,
  KHR_500: 500,
  KHR_1000: 1000,
  KHR_2000: 2000,
  KHR_5000: 5000,
  KHR_10000: 10000,
  KHR_20000: 20000,
  KHR_50000: 50000,
};
const COUNTING_MODE = "post_nms_detector_proposals_with_fragment_class_override";
const COUNT_SOURCE = "final_nms_detections";
const COLORS = [
  "#e43d30",
  "#2478c2",
  "#f0a202",
  "#1b998b",
  "#9b5de5",
  "#ef476f",
  "#06d6a0",
  "#ffd166",
  "#118ab2",
  "#f78c6b",
  "#00b4d8",
  "#f72585",
  "#90be6d",
];

const state = {
  config: null,
  detectorSession: null,
  classifierSession: null,
  image: null,
  detections: [],
  conf: 0.05,
  autoRunRequested: false,
  autoRunDone: false,
  debug: {},
};

const canvas = document.getElementById("imageCanvas");
const ctx = canvas.getContext("2d");
const imageInput = document.getElementById("imageInput");
const runButton = document.getElementById("runButton");
const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");
const modelStatus = document.getElementById("modelStatus");
const emptyState = document.getElementById("emptyState");
const totalCount = document.getElementById("totalCount");
const totalValue = document.getElementById("totalValue");
const countsList = document.getElementById("countsList");

async function loadModel() {
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    const configResponse = await fetch(stackConfigUrl());
    if (!configResponse.ok) {
      throw new Error(`Config HTTP ${configResponse.status}`);
    }
    state.config = await configResponse.json();
    applyConfigOverrides();
    state.conf = state.config.detector.proposal_confidence ?? state.conf;
    confSlider.value = String(state.conf);
    confValue.textContent = state.conf.toFixed(2);
    const options = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    };
    [state.detectorSession, state.classifierSession] = await Promise.all([
      ort.InferenceSession.create(repoUrl(state.config.detector.path), options),
      ort.InferenceSession.create(repoUrl(state.config.fragment_classifier.path), options),
    ]);
    modelStatus.textContent = "Models ready";
    modelStatus.className = "status ready";
    updateRunState();
    maybeAutoRun();
  } catch (error) {
    modelStatus.textContent = "Model load failed";
    modelStatus.className = "status error";
    console.error(error);
  }
}

function queryNumber(name) {
  const raw = params.get(name);
  if (raw === null || raw.trim() === "") {
    return null;
  }
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function queryFlag(name) {
  const raw = params.get(name);
  if (raw === null) {
    return false;
  }
  return !["", "0", "false", "no", "off"].includes(raw.trim().toLowerCase());
}

function queryText(...names) {
  for (const name of names) {
    const raw = params.get(name);
    if (raw !== null && raw.trim() !== "") {
      return raw.trim();
    }
  }
  return "";
}

function stackConfigUrl() {
  const override = queryText("stackConfig", "config");
  return override ? repoUrl(override) : STACK_CONFIG_URL;
}

function applyConfigOverrides() {
  const proposalConf = queryNumber("proposalConf") ?? queryNumber("conf");
  const detectorOverride = queryNumber("detectorOverride");
  const nmsIou = queryNumber("nmsIou");
  const cropPadding = queryNumber("cropPadding");
  const detectorModel = queryText("detectorModel", "detectorPath");
  const fragmentClassifierModel = queryText("fragmentClassifierModel", "fragmentModel", "fragmentPath");
  const fragmentDisagreementMinConf = queryNumber("fragmentDisagreementMinConf");
  const unclassifiedMinConf = queryNumber("unclassifiedMinConf");
  if (proposalConf !== null) {
    state.config.detector.proposal_confidence = proposalConf;
  }
  if (detectorModel) {
    state.config.detector.path = detectorModel;
  }
  if (fragmentClassifierModel) {
    state.config.fragment_classifier.path = fragmentClassifierModel;
  }
  if (detectorOverride !== null) {
    state.config.fusion.detector_override_confidence = detectorOverride;
  }
  if (queryFlag("rejectFragmentDisagreement")) {
    state.config.fusion.reject_fragment_disagreement = true;
  }
  if (fragmentDisagreementMinConf !== null) {
    state.config.fusion.reject_fragment_disagreement_min_conf = fragmentDisagreementMinConf;
  }
  if (unclassifiedMinConf !== null) {
    state.config.fusion.unclassified_min_conf = unclassifiedMinConf;
  }
  if (nmsIou !== null) {
    state.config.fusion.nms_iou = nmsIou;
  }
  if (cropPadding !== null) {
    state.config.fragment_classifier.crop_padding = cropPadding;
  }
}

function updateRunState() {
  runButton.disabled = !state.detectorSession || !state.classifierSession || !state.image;
}

function maybeAutoRun() {
  if (state.autoRunRequested && !state.autoRunDone && state.detectorSession && state.classifierSession && state.image) {
    state.autoRunDone = true;
    runModel();
  }
}

function repoUrl(path) {
  return `/${path.replaceAll("\\", "/").replace(/^\/+/, "")}`;
}

function sourceImageData(image) {
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = image.width;
  sourceCanvas.height = image.height;
  const sourceCtx = sourceCanvas.getContext("2d", { willReadFrequently: true });
  sourceCtx.drawImage(image, 0, 0);
  return sourceCtx.getImageData(0, 0, image.width, image.height).data;
}

function writeResizedChannelFirst(input, image, sourcePixels, inputSize, resizedWidth, resizedHeight, padX, padY) {
  const sourceXScale = image.width / resizedWidth;
  const sourceYScale = image.height / resizedHeight;
  const channelStride = inputSize * inputSize;
  input.fill(114 / 255);
  for (let y = 0; y < resizedHeight; y += 1) {
    const sourceY = Math.max(0, (y + 0.5) * sourceYScale - 0.5);
    const y0 = Math.floor(sourceY);
    const y1 = Math.min(image.height - 1, y0 + 1);
    const wy = sourceY - y0;
    for (let x = 0; x < resizedWidth; x += 1) {
      const sourceX = Math.max(0, (x + 0.5) * sourceXScale - 0.5);
      const x0 = Math.floor(sourceX);
      const x1 = Math.min(image.width - 1, x0 + 1);
      const wx = sourceX - x0;
      const topOffset = (y0 * image.width + x0) * 4;
      const topRightOffset = (y0 * image.width + x1) * 4;
      const bottomOffset = (y1 * image.width + x0) * 4;
      const bottomRightOffset = (y1 * image.width + x1) * 4;
      const target = (y + padY) * inputSize + x + padX;
      for (let channel = 0; channel < 3; channel += 1) {
        const top = sourcePixels[topOffset + channel] * (1 - wx) + sourcePixels[topRightOffset + channel] * wx;
        const bottom = sourcePixels[bottomOffset + channel] * (1 - wx) + sourcePixels[bottomRightOffset + channel] * wx;
        input[target + channel * channelStride] = (top * (1 - wy) + bottom * wy) / 255;
      }
    }
  }
}

function drawBaseImage() {
  if (!state.image) {
    return;
  }
  const maxSide = 1800;
  const scale = Math.min(1, maxSide / Math.max(state.image.width, state.image.height));
  canvas.width = Math.round(state.image.width * scale);
  canvas.height = Math.round(state.image.height * scale);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
}

function preprocess(image) {
  const inputSize = state.config?.detector?.input_size ?? 416;
  const scale = Math.min(inputSize / image.width, inputSize / image.height);
  const resizedWidth = Math.round(image.width * scale);
  const resizedHeight = Math.round(image.height * scale);
  const padX = Math.floor((inputSize - resizedWidth) / 2);
  const padY = Math.floor((inputSize - resizedHeight) / 2);
  const input = new Float32Array(3 * inputSize * inputSize);
  writeResizedChannelFirst(input, image, sourceImageData(image), inputSize, resizedWidth, resizedHeight, padX, padY);

  return {
    tensor: new ort.Tensor("float32", input, [1, 3, inputSize, inputSize]),
    scale,
    padX,
    padY,
  };
}

function parseOutput(output, meta) {
  const detections = [];
  for (let i = 0; i < output.dims[1]; i += 1) {
    const offset = i * 6;
    const score = output.data[offset + 4];
    if (score < state.conf) {
      continue;
    }
    const classId = Math.round(output.data[offset + 5]);
    const classNames = state.config?.detector?.classes ?? DEFAULT_CLASS_NAMES;
    const x1 = (output.data[offset] - meta.padX) / meta.scale;
    const y1 = (output.data[offset + 1] - meta.padY) / meta.scale;
    const x2 = (output.data[offset + 2] - meta.padX) / meta.scale;
    const y2 = (output.data[offset + 3] - meta.padY) / meta.scale;
    detections.push({
      classId,
      detectorName: classNames[classId] || `class_${classId}`,
      detectorScore: score,
      fragmentName: "",
      fragmentScore: 0,
      name: classNames[classId] || `class_${classId}`,
      score,
      x1: clamp(x1, 0, state.image.width),
      y1: clamp(y1, 0, state.image.height),
      x2: clamp(x2, 0, state.image.width),
      y2: clamp(y2, 0, state.image.height),
    });
  }
  return detections;
}

function preprocessClassifierBatch(image, detections) {
  const size = state.config?.fragment_classifier?.input_size ?? 224;
  const area = size * size;
  const input = new Float32Array(detections.length * 3 * area);
  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = size;
  cropCanvas.height = size;
  const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });
  const mean = state.config?.fragment_classifier?.normalization?.mean ?? [0.485, 0.456, 0.406];
  const std = state.config?.fragment_classifier?.normalization?.std ?? [0.229, 0.224, 0.225];

  detections.forEach((detection, batchIndex) => {
    const boxWidth = detection.x2 - detection.x1;
    const boxHeight = detection.y2 - detection.y1;
    const cropPadding = state.config?.fragment_classifier?.crop_padding ?? 0.08;
    const padX = boxWidth * cropPadding;
    const padY = boxHeight * cropPadding;
    const sx = clamp(detection.x1 - padX, 0, image.width);
    const sy = clamp(detection.y1 - padY, 0, image.height);
    const ex = clamp(detection.x2 + padX, 0, image.width);
    const ey = clamp(detection.y2 + padY, 0, image.height);

    cropCtx.fillStyle = "white";
    cropCtx.fillRect(0, 0, size, size);
    cropCtx.drawImage(image, sx, sy, Math.max(1, ex - sx), Math.max(1, ey - sy), 0, 0, size, size);
    const pixels = cropCtx.getImageData(0, 0, size, size).data;
    const batchOffset = batchIndex * 3 * area;
    for (let i = 0; i < area; i += 1) {
      input[batchOffset + i] = (pixels[i * 4] / 255 - mean[0]) / std[0];
      input[batchOffset + area + i] = (pixels[i * 4 + 1] / 255 - mean[1]) / std[1];
      input[batchOffset + area * 2 + i] = (pixels[i * 4 + 2] / 255 - mean[2]) / std[2];
    }
  });

  return new ort.Tensor("float32", input, [detections.length, 3, size, size]);
}

function softmaxRow(data, offset, count) {
  let maxLogit = -Infinity;
  for (let i = 0; i < count; i += 1) {
    maxLogit = Math.max(maxLogit, data[offset + i]);
  }
  let sum = 0;
  const probs = [];
  for (let i = 0; i < count; i += 1) {
    const value = Math.exp(data[offset + i] - maxLogit);
    probs.push(value);
    sum += value;
  }
  return probs.map((value) => value / sum);
}

async function classifyFragments(detections) {
  if (!detections.length) {
    return detections;
  }
  const fragmentClassNames = state.config?.fragment_classifier?.classes ?? ["KHR_1000", "KHR_10000", "KHR_20000", "KHR_5000"];
  const applyToAllProposals = state.config?.fragment_classifier?.apply_to_all_proposals ?? false;
  const eligible = detections
    .map((detection, index) => ({ detection, index }))
    .filter((item) => applyToAllProposals || fragmentClassNames.includes(item.detection.detectorName));
  if (!eligible.length) {
    return detections;
  }
  const tensor = preprocessClassifierBatch(
    state.image,
    eligible.map((item) => item.detection),
  );
  const feeds = { [state.classifierSession.inputNames[0]]: tensor };
  const outputs = await state.classifierSession.run(feeds);
  const logits = outputs[state.classifierSession.outputNames[0]];
  const classCount = fragmentClassNames.length;
  const classified = [...detections];
  eligible.forEach(({ detection, index }, batchIndex) => {
    const probs = softmaxRow(logits.data, batchIndex * classCount, classCount);
    let bestIndex = 0;
    for (let i = 1; i < probs.length; i += 1) {
      if (probs[i] > probs[bestIndex]) {
        bestIndex = i;
      }
    }
    const fragmentName = fragmentClassNames[bestIndex];
    const fragmentScore = probs[bestIndex];
    const overrideConf = state.config?.fusion?.detector_override_confidence ?? 0.17;
    const rejectOnDisagreement = state.config?.fusion?.reject_fragment_disagreement ?? false;
    const disagreementMinConf = state.config?.fusion?.reject_fragment_disagreement_min_conf ?? 0;
    const rejectClasses = new Set(state.config?.fusion?.reject_fragment_classes ?? []);
    const rejectClassMinConf = state.config?.fusion?.reject_fragment_class_min_conf ?? 0;
    const rejectedByClass = rejectClasses.has(fragmentName) && fragmentScore >= rejectClassMinConf;
    const rejectedByDisagreement =
      rejectOnDisagreement && fragmentName !== detection.detectorName && fragmentScore >= disagreementMinConf;
    const rejected = rejectedByClass || rejectedByDisagreement;
    const fragmentOverrideEnabled = state.config?.fusion?.fragment_override_enabled ?? true;
    const useDetector = !fragmentOverrideEnabled || detection.detectorScore >= overrideConf;
    classified[index] = {
      ...detection,
      fragmentName,
      fragmentScore,
      rejected,
      rejectReason: rejectedByClass ? `fragment_class:${fragmentName}` : rejectedByDisagreement ? "fragment_disagreement" : "",
      name: useDetector ? detection.detectorName : fragmentName,
      score: useDetector ? detection.detectorScore : fragmentScore,
    };
  });
  return classified;
}

function boxIou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = areaA + areaB - intersection;
  return union ? intersection / union : 0;
}

function nms(detections) {
  const nmsIou = state.config?.fusion?.nms_iou ?? 0.85;
  const pending = [...detections].sort((a, b) => b.detectorScore - a.detectorScore);
  const kept = [];
  for (const detection of pending) {
    if (kept.every((keptDetection) => boxIou(detection, keptDetection) < nmsIou)) {
      kept.push(detection);
    }
  }
  return kept.sort((a, b) => b.score - a.score);
}

function diagnosticDetection(detection) {
  return {
    classId: detection.classId,
    detectorName: detection.detectorName || "",
    detectorScore: Number(detection.detectorScore || 0),
    fragmentName: detection.fragmentName || "",
    fragmentScore: detection.fragmentScore == null ? null : Number(detection.fragmentScore || 0),
    name: detection.name || "",
    score: Number(detection.score || 0),
    x1: detection.x1,
    y1: detection.y1,
    x2: detection.x2,
    y2: detection.y2,
    rejected: Boolean(detection.rejected),
    rejectReason: detection.rejectReason || "",
  };
}

function diagnosticDetections(detections) {
  return detections.map((detection) => diagnosticDetection(detection));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function renderDetections() {
  drawBaseImage();
  const sx = canvas.width / state.image.width;
  const sy = canvas.height / state.image.height;
  ctx.lineWidth = Math.max(3, Math.round(canvas.width / 420));
  ctx.font = `${Math.max(18, Math.round(canvas.width / 42))}px Aptos, Segoe UI, sans-serif`;
  ctx.textBaseline = "top";

  for (const detection of state.detections) {
    const color = COLORS[detection.classId % COLORS.length];
    const x = detection.x1 * sx;
    const y = detection.y1 * sy;
    const w = (detection.x2 - detection.x1) * sx;
    const h = (detection.y2 - detection.y1) * sy;
    const label = `${detection.name} ${detection.score.toFixed(2)}`;
    const textWidth = ctx.measureText(label).width + 12;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.strokeRect(x, y, w, h);
    ctx.fillRect(x, Math.max(0, y - 34), textWidth, 34);
    ctx.fillStyle = "#111";
    ctx.fillText(label, x + 6, Math.max(0, y - 30));
  }
  renderSummary();
}

function renderSummary() {
  const counts = new Map();
  let khrValue = 0;
  let usdValue = 0;
  for (const detection of state.detections) {
    counts.set(detection.name, (counts.get(detection.name) || 0) + 1);
    if (detection.name.startsWith("USD_")) {
      usdValue += VALUES[detection.name] || 0;
    } else if (detection.name.startsWith("KHR_")) {
      khrValue += VALUES[detection.name] || 0;
    }
  }
  totalCount.textContent = String(state.detections.length);
  totalValue.innerHTML = `<span class="currency-line">KHR ${khrValue.toLocaleString()}</span><span class="currency-line">USD ${usdValue.toLocaleString()}</span>`;
  countsList.innerHTML = "";
  for (const [name, count] of [...counts.entries()].sort()) {
    const classNames = state.config?.detector?.classes ?? DEFAULT_CLASS_NAMES;
    const classId = classNames.indexOf(name);
    const row = document.createElement("div");
    row.className = "count-row";
    row.innerHTML = `<span><i class="swatch" style="background:${COLORS[classId % COLORS.length]}"></i>${name}</span><strong>${count}</strong>`;
    countsList.append(row);
  }
}

function setLoadedImage(image) {
  state.image = image;
  state.detections = [];
  emptyState.classList.add("hidden");
  drawBaseImage();
  renderSummary();
  updateRunState();
  maybeAutoRun();
}

function loadImageUrl(path) {
  const image = new Image();
  image.onload = () => setLoadedImage(image);
  image.onerror = () => {
    emptyState.textContent = `Image load failed: ${path}`;
  };
  image.src = repoUrl(path);
}

async function runModel() {
  if (!state.detectorSession || !state.classifierSession || !state.image) {
    return;
  }
  runButton.disabled = true;
  runButton.textContent = "Running";
  const meta = preprocess(state.image);
  const feeds = { [state.detectorSession.inputNames[0]]: meta.tensor };
  const outputs = await state.detectorSession.run(feeds);
  const output = outputs[state.detectorSession.outputNames[0]];
  const proposals = parseOutput(output, meta);
  const classified = await classifyFragments(proposals);
  const unclassifiedMinConf = state.config?.fusion?.unclassified_min_conf ?? 0;
  const rejectAfterNms = state.config?.fusion?.reject_after_nms ?? false;
  const filtered = classified.map((detection) => {
    if (!detection.fragmentName && unclassifiedMinConf > 0 && detection.detectorScore < unclassifiedMinConf) {
      return { ...detection, rejected: true, rejectReason: "unclassified_low_conf" };
    }
    return detection;
  });
  const clustered = rejectAfterNms ? nms(filtered) : nms(filtered.filter((detection) => !detection.rejected));
  state.detections = clustered.filter((detection) => !detection.rejected);
  const fragmentClassified = filtered.filter((detection) => detection.fragmentName).length;
  const rejectedProposals = filtered.filter((detection) => detection.rejected).length;
  const rejectedAfterNms = clustered.filter((detection) => detection.rejected).length;
  state.debug = {
    countingMode: COUNTING_MODE,
    countSource: COUNT_SOURCE,
    detectorOutputDims: output.dims,
    detectorProposals: proposals.length,
    classifiedProposals: filtered.length,
    fragmentClassifiedProposals: fragmentClassified,
    rejectedProposals,
    fragmentDisagreementReject: state.config?.fusion?.reject_fragment_disagreement ?? false,
    fragmentClassifierApplyToAll: state.config?.fragment_classifier?.apply_to_all_proposals ?? false,
    rejectFragmentClasses: state.config?.fusion?.reject_fragment_classes ?? [],
    rejectFragmentClassMinConf: state.config?.fusion?.reject_fragment_class_min_conf ?? 0,
    fragmentOverrideEnabled: state.config?.fusion?.fragment_override_enabled ?? true,
    rejectAfterNms,
    clusteredProposals: clustered.length,
    rejectedAfterNms,
    proposalDetections: diagnosticDetections(proposals),
    classifiedDetections: diagnosticDetections(filtered),
    clusteredDetections: diagnosticDetections(clustered),
    unclassifiedMinConf,
    finalDetections: state.detections.length,
    proposals: proposals.length,
    classified: filtered.length,
    fragmentClassified,
    rejected: rejectedProposals,
    final: state.detections.length,
  };
  renderDetections();
  runButton.textContent = "Run";
  updateRunState();
}

imageInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    URL.revokeObjectURL(url);
    state.autoRunRequested = false;
    setLoadedImage(image);
  };
  image.src = url;
});

confSlider.addEventListener("input", () => {
  state.conf = Number(confSlider.value);
  confValue.textContent = state.conf.toFixed(2);
});

runButton.addEventListener("click", runModel);

const params = new URLSearchParams(window.location.search);
const sampleImage = params.get("image");
if (params.get("autorun") === "1") {
  state.autoRunRequested = true;
}
if (sampleImage) {
  loadImageUrl(sampleImage);
}

loadModel();
