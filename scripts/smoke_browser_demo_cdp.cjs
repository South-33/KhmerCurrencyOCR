const fs = require("node:fs");
const path = require("node:path");
const { spawn, spawnSync } = require("node:child_process");

const ROOT = path.resolve(__dirname, "..");
const DEFAULT_EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const CLASS_NAMES = [
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

function parseArgs(argv) {
  const args = {
    image: "/data/real_fan_benchmark/images/candidates/real_overlap_0003_commons_shop_5k_10k_20k.png",
    port: 8787,
    debugPort: 9223,
    timeoutMs: 120000,
    screenshot: "",
    outCsv: "",
    outJson: "",
    labels: "",
    matchIou: 0.5,
    proposalConf: "",
    detectorOverride: "",
    detectorModel: "",
    fragmentClassifierModel: "",
    stackConfig: "",
    rejectFragmentDisagreement: false,
    fragmentDisagreementMinConf: "",
    unclassifiedMinConf: "",
    nmsIou: "",
    cropPadding: "",
    minSameClass: null,
    minAnyClass: null,
    maxCountError: null,
    maxKhrError: null,
    maxUsdError: null,
    edge: process.env.EDGE_PATH || DEFAULT_EDGE,
  };
  for (let index = 2; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (key === "--image") {
      args.image = value;
      index += 1;
    } else if (key === "--port") {
      args.port = Number(value);
      index += 1;
    } else if (key === "--debug-port") {
      args.debugPort = Number(value);
      index += 1;
    } else if (key === "--timeout-ms") {
      args.timeoutMs = Number(value);
      index += 1;
    } else if (key === "--screenshot") {
      args.screenshot = value;
      index += 1;
    } else if (key === "--out-csv") {
      args.outCsv = value;
      index += 1;
    } else if (key === "--out-json") {
      args.outJson = value;
      index += 1;
    } else if (key === "--labels") {
      args.labels = value;
      index += 1;
    } else if (key === "--match-iou") {
      args.matchIou = Number(value);
      index += 1;
    } else if (key === "--proposal-conf") {
      args.proposalConf = value;
      index += 1;
    } else if (key === "--detector-override") {
      args.detectorOverride = value;
      index += 1;
    } else if (key === "--detector-model" || key === "--detector-path") {
      args.detectorModel = value;
      index += 1;
    } else if (
      key === "--fragment-classifier-model" ||
      key === "--fragment-classifier-path" ||
      key === "--fragment-model" ||
      key === "--fragment-path"
    ) {
      args.fragmentClassifierModel = value;
      index += 1;
    } else if (key === "--stack-config" || key === "--config") {
      args.stackConfig = value;
      index += 1;
    } else if (key === "--reject-fragment-disagreement") {
      args.rejectFragmentDisagreement = true;
    } else if (key === "--fragment-disagreement-min-conf") {
      args.fragmentDisagreementMinConf = value;
      index += 1;
    } else if (key === "--unclassified-min-conf") {
      args.unclassifiedMinConf = value;
      index += 1;
    } else if (key === "--nms-iou") {
      args.nmsIou = value;
      index += 1;
    } else if (key === "--crop-padding") {
      args.cropPadding = value;
      index += 1;
    } else if (key === "--min-same-class") {
      args.minSameClass = Number(value);
      index += 1;
    } else if (key === "--min-any-class") {
      args.minAnyClass = Number(value);
      index += 1;
    } else if (key === "--max-count-error") {
      args.maxCountError = Number(value);
      index += 1;
    } else if (key === "--max-khr-error") {
      args.maxKhrError = Number(value);
      index += 1;
    } else if (key === "--max-usd-error") {
      args.maxUsdError = Number(value);
      index += 1;
    } else if (key === "--edge") {
      args.edge = value;
      index += 1;
    } else if (key === "--help" || key === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${key}`);
    }
  }
  return args;
}

function printHelp() {
  console.log(`Usage: node scripts/smoke_browser_demo_cdp.cjs [options]

Options:
  --image PATH        Repo-root browser image URL path.
  --labels PATH       Optional YOLO detect labels for smoke evaluation.
  --match-iou NUMBER  IoU threshold for --labels evaluation. Default: 0.5.
  --proposal-conf N   Override browser detector proposal confidence.
  --detector-override N Override detector-vs-fragment threshold.
  --detector-model PATH Override detector ONNX model path served from the repo root.
  --fragment-classifier-model PATH Override fragment classifier ONNX model path served from the repo root.
  --stack-config PATH Override stack config JSON served from the repo root.
  --reject-fragment-disagreement Reject proposals when detector and fragment classifier disagree.
  --fragment-disagreement-min-conf N Minimum fragment confidence for disagreement rejection.
  --unclassified-min-conf N Reject proposals below N when they are not fragment-classified.
  --nms-iou N         Override browser fusion NMS IoU.
  --crop-padding N    Override fragment crop padding fraction.
  --min-same-class N  Fail if labeled final same-class matches are below N.
  --min-any-class N   Fail if labeled final any-class matches are below N.
  --max-count-error N Fail if absolute final count error is above N. Requires --labels.
  --max-khr-error N   Fail if absolute KHR value error is above N. Requires --labels.
  --max-usd-error N   Fail if absolute USD value error is above N. Requires --labels.
  --screenshot PATH   Optional PNG screenshot output path.
  --out-csv PATH      Optional CSV output for browser-side detections.
  --out-json PATH     Optional JSON output for the smoke summary.
  --port NUMBER       Local static server port. Default: 8787.
  --debug-port NUMBER Edge DevTools port. Default: 9223.
  --timeout-ms NUMBER Autorun wait timeout. Default: 120000.
  --edge PATH         Microsoft Edge executable path.`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function killTree(child) {
  if (!child || !child.pid) return;
  if (process.platform === "win32") {
    spawnSync("powershell", ["-NoProfile", "-Command", `Stop-Process -Id ${child.pid} -Force -ErrorAction SilentlyContinue`], {
      stdio: "ignore",
      windowsHide: true,
    });
  } else {
    child.kill("SIGTERM");
  }
}

function stopWindowsProcessesByCommandLine(pattern) {
  if (process.platform !== "win32") return;
  const escaped = pattern.replaceAll("'", "''");
  spawnSync(
    "powershell",
    [
      "-NoProfile",
      "-Command",
      `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*${escaped}*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }`,
    ],
    { stdio: "ignore", windowsHide: true },
  );
}

async function waitForHttp(url, timeoutMs) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // Server is still starting.
    }
    await sleep(250);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

async function connectCdp(debugPort, timeoutMs) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const pages = await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json();
      const page = pages.find((item) => (item.url || "").includes("/demo/browser/")) || pages[0];
      if (page?.webSocketDebuggerUrl) {
        const ws = new WebSocket(page.webSocketDebuggerUrl);
        await new Promise((resolve, reject) => {
          ws.onopen = resolve;
          ws.onerror = reject;
        });
        return ws;
      }
    } catch {
      // Edge is still starting.
    }
    await sleep(250);
  }
  throw new Error(`Timed out waiting for Edge CDP on port ${debugPort}`);
}

function createCdpClient(ws) {
  let id = 0;
  const pending = new Map();
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    }
  };
  return function send(method, params = {}) {
    const messageId = ++id;
    ws.send(JSON.stringify({ id: messageId, method, params }));
    return new Promise((resolve) => pending.set(messageId, resolve));
  };
}

function readYoloLabels(labelsPath, imageWidth, imageHeight) {
  if (!labelsPath) return null;
  const absolute = path.resolve(ROOT, labelsPath);
  const text = fs.readFileSync(absolute, "utf8");
  const labels = [];
  text.split(/\r?\n/).forEach((line, lineIndex) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) return;
    const parts = trimmed.split(/\s+/).map(Number);
    if (parts.length !== 5 || parts.some((part) => Number.isNaN(part))) {
      throw new Error(`${labelsPath}: line ${lineIndex + 1} must be YOLO detect format`);
    }
    const [classId, cx, cy, width, height] = parts;
    labels.push({
      classId,
      name: CLASS_NAMES[classId] || `class_${classId}`,
      x1: (cx - width / 2) * imageWidth,
      y1: (cy - height / 2) * imageHeight,
      x2: (cx + width / 2) * imageWidth,
      y2: (cy + height / 2) * imageHeight,
    });
  });
  return labels;
}

function boxIou(left, right) {
  const x1 = Math.max(left.x1, right.x1);
  const y1 = Math.max(left.y1, right.y1);
  const x2 = Math.min(left.x2, right.x2);
  const y2 = Math.min(left.y2, right.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const leftArea = Math.max(0, left.x2 - left.x1) * Math.max(0, left.y2 - left.y1);
  const rightArea = Math.max(0, right.x2 - right.x1) * Math.max(0, right.y2 - right.y1);
  const union = leftArea + rightArea - intersection;
  return union ? intersection / union : 0;
}

function greedyMatchPairs(detections, labels, matchIou, requireSameClass) {
  const usedLabels = new Set();
  const matches = [];
  [...detections].sort((left, right) => Number(right.score || 0) - Number(left.score || 0)).forEach((detection) => {
    let bestIndex = -1;
    let bestIou = 0;
    labels.forEach((label, index) => {
      if (usedLabels.has(index)) return;
      if (requireSameClass && detection.name !== label.name) return;
      const iou = boxIou(detection, label);
      if (iou > bestIou) {
        bestIou = iou;
        bestIndex = index;
      }
    });
    if (bestIndex >= 0 && bestIou >= matchIou) {
      usedLabels.add(bestIndex);
      matches.push({ detection, label: labels[bestIndex], iou: bestIou });
    }
  });
  return matches;
}

function confusionCounts(matches) {
  const counts = {};
  for (const match of matches) {
    if (match.label.name === match.detection.name) continue;
    const key = `${match.label.name}->${match.detection.name}`;
    counts[key] = (counts[key] || 0) + 1;
  }
  return Object.fromEntries(Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)));
}

function matchRows(matches) {
  return matches.map((match) => ({
    target: match.label.name,
    prediction: match.detection.name,
    iou: Number(match.iou.toFixed(4)),
    score: Number(Number(match.detection.score || 0).toFixed(4)),
  }));
}

function sourceDetections(rawDetections, nameKey, scoreKey) {
  return rawDetections
    .filter((detection) => detection[nameKey])
    .map((detection) => ({
      ...detection,
      name: detection[nameKey],
      score: Number(detection[scoreKey] || 0),
    }));
}

function evaluateSource(detections, labels, matchIou) {
  const sameClassPairs = greedyMatchPairs(detections, labels, matchIou, true);
  const anyClassPairs = greedyMatchPairs(detections, labels, matchIou, false);
  const matchedSameClass = sameClassPairs.length;
  const matchedAnyClass = anyClassPairs.length;
  return {
    predCount: detections.length,
    countError: detections.length - labels.length,
    matchedSameClass,
    matchedAnyClass,
    recallSameClass: labels.length ? matchedSameClass / labels.length : 0,
    recallAnyClass: labels.length ? matchedAnyClass / labels.length : 0,
    confusions: confusionCounts(anyClassPairs),
    matchedPairs: matchRows(anyClassPairs),
  };
}

function currencyValues(items, nameKey = "name") {
  const values = { khrValue: 0, usdValue: 0 };
  for (const item of items || []) {
    const name = item[nameKey];
    if (name?.startsWith("KHR_")) {
      values.khrValue += VALUES[name] || 0;
    } else if (name?.startsWith("USD_")) {
      values.usdValue += VALUES[name] || 0;
    }
  }
  return values;
}

function evaluateDetections(value, args) {
  const labels = readYoloLabels(args.labels, value.imageWidth, value.imageHeight);
  if (!labels) return null;
  const detections = value.detections || [];
  const debug = value.debug || {};
  const proposalDetections = Array.isArray(debug.proposalDetections) ? debug.proposalDetections : [];
  const classifiedDetections = Array.isArray(debug.classifiedDetections) ? debug.classifiedDetections : [];
  const clusteredDetections = Array.isArray(debug.clusteredDetections) ? debug.clusteredDetections : [];
  const expected = currencyValues(labels);
  const predicted = currencyValues(detections);
  const sources = {
    final: evaluateSource(sourceDetections(detections, "name", "score"), labels, args.matchIou),
    detector: evaluateSource(sourceDetections(detections, "detectorName", "detectorScore"), labels, args.matchIou),
    fragment: evaluateSource(sourceDetections(detections, "fragmentName", "fragmentScore"), labels, args.matchIou),
  };
  if (proposalDetections.length) {
    sources.proposals_detector = evaluateSource(
      sourceDetections(proposalDetections, "detectorName", "detectorScore"),
      labels,
      args.matchIou,
    );
  }
  if (classifiedDetections.length) {
    sources.classified_detector = evaluateSource(
      sourceDetections(classifiedDetections, "detectorName", "detectorScore"),
      labels,
      args.matchIou,
    );
    sources.classified_fragment = evaluateSource(
      sourceDetections(classifiedDetections, "fragmentName", "fragmentScore"),
      labels,
      args.matchIou,
    );
  }
  if (clusteredDetections.length) {
    sources.clustered_detector = evaluateSource(
      sourceDetections(clusteredDetections, "detectorName", "detectorScore"),
      labels,
      args.matchIou,
    );
    sources.clustered_final = evaluateSource(sourceDetections(clusteredDetections, "name", "score"), labels, args.matchIou);
    sources.clustered_fragment = evaluateSource(
      sourceDetections(clusteredDetections, "fragmentName", "fragmentScore"),
      labels,
      args.matchIou,
    );
  }
  return {
    labels: path.relative(ROOT, path.resolve(ROOT, args.labels)),
    matchIou: args.matchIou,
    gtCount: labels.length,
    gtKhrValue: expected.khrValue,
    gtUsdValue: expected.usdValue,
    khrValueError: predicted.khrValue - expected.khrValue,
    usdValueError: predicted.usdValue - expected.usdValue,
    ...sources.final,
    sources,
  };
}

function summarize(value, args) {
  const counts = {};
  for (const detection of value.detections || []) {
    counts[detection.name] = (counts[detection.name] || 0) + 1;
  }
  const values = currencyValues(value.detections || []);
  const totalCount = Number(value.totalCount || 0);
  const detections = value.detections || [];
  const debug = value.debug || {};
  const predClassTotal = Object.values(counts).reduce((total, count) => total + count, 0);
  return {
    status: value.status,
    totalCount,
    ...values,
    predClasses: Object.fromEntries(Object.entries(counts).sort(([left], [right]) => left.localeCompare(right))),
    countContract: {
      mode: debug.countingMode || "",
      expectedMode: COUNTING_MODE,
      countSource: debug.countSource || "",
      expectedCountSource: COUNT_SOURCE,
      totalCount,
      finalDetections: detections.length,
      predClassTotal,
      detectorProposals: Number(debug.detectorProposals ?? debug.proposals ?? 0),
      classifiedProposals: Number(debug.classifiedProposals ?? debug.classified ?? 0),
      fragmentClassifiedProposals: Number(debug.fragmentClassifiedProposals ?? debug.fragmentClassified ?? 0),
      uiTotalMatchesFinal: totalCount === detections.length,
      predClassTotalMatchesFinal: predClassTotal === detections.length,
      debugFinalMatchesFinal: Number(debug.finalDetections ?? debug.final ?? -1) === detections.length,
      finalNotMoreThanClassified: detections.length <= Number(debug.classifiedProposals ?? debug.classified ?? -1),
      fragmentClassifiedNotMoreThanClassified:
        Number(debug.fragmentClassifiedProposals ?? debug.fragmentClassified ?? -1) <=
        Number(debug.classifiedProposals ?? debug.classified ?? -1),
    },
    debug,
    evaluation: evaluateDetections(value, args),
    detections,
  };
}

function enforceCountContract(summary) {
  const contract = summary.countContract || {};
  const failures = [];
  if (contract.mode !== COUNTING_MODE) {
    failures.push(`mode ${contract.mode || "<missing>"} != ${COUNTING_MODE}`);
  }
  if (contract.countSource !== COUNT_SOURCE) {
    failures.push(`count source ${contract.countSource || "<missing>"} != ${COUNT_SOURCE}`);
  }
  if (!contract.uiTotalMatchesFinal) {
    failures.push(`UI total ${contract.totalCount} != final detections ${contract.finalDetections}`);
  }
  if (!contract.predClassTotalMatchesFinal) {
    failures.push(`class total ${contract.predClassTotal} != final detections ${contract.finalDetections}`);
  }
  if (!contract.debugFinalMatchesFinal) {
    failures.push(`debug final != final detections ${contract.finalDetections}`);
  }
  if (!contract.finalNotMoreThanClassified) {
    failures.push(
      `final detections ${contract.finalDetections} > classified proposals ${contract.classifiedProposals}`,
    );
  }
  if (!contract.fragmentClassifiedNotMoreThanClassified) {
    failures.push(
      `fragment-classified proposals ${contract.fragmentClassifiedProposals} > classified proposals ${contract.classifiedProposals}`,
    );
  }
  if (failures.length) {
    throw new Error(`Browser smoke count contract failed: ${failures.join("; ")}`);
  }
}

function enforceEvaluation(summary, args) {
  if (
    args.minSameClass === null &&
    args.minAnyClass === null &&
    args.maxCountError === null &&
    args.maxKhrError === null &&
    args.maxUsdError === null
  ) {
    return;
  }
  if (!summary.evaluation) {
    throw new Error("--min-same-class/--min-any-class/--max-count-error/--max-khr-error/--max-usd-error require --labels");
  }
  const failures = [];
  if (args.minSameClass !== null && summary.evaluation.matchedSameClass < args.minSameClass) {
    failures.push(`same-class ${summary.evaluation.matchedSameClass} < ${args.minSameClass}`);
  }
  if (args.minAnyClass !== null && summary.evaluation.matchedAnyClass < args.minAnyClass) {
    failures.push(`any-class ${summary.evaluation.matchedAnyClass} < ${args.minAnyClass}`);
  }
  if (args.maxCountError !== null && Math.abs(summary.evaluation.countError) > args.maxCountError) {
    failures.push(`count error ${summary.evaluation.countError} exceeds +/-${args.maxCountError}`);
  }
  if (args.maxKhrError !== null && Math.abs(summary.evaluation.khrValueError) > args.maxKhrError) {
    failures.push(`KHR value error ${summary.evaluation.khrValueError} exceeds +/-${args.maxKhrError}`);
  }
  if (args.maxUsdError !== null && Math.abs(summary.evaluation.usdValueError) > args.maxUsdError) {
    failures.push(`USD value error ${summary.evaluation.usdValueError} exceeds +/-${args.maxUsdError}`);
  }
  if (failures.length) {
    throw new Error(`Browser smoke evaluation failed: ${failures.join("; ")}`);
  }
}

async function readBrowserState(send, timeoutMs) {
  await send("Runtime.enable");
  const expression = `(() => JSON.stringify({
    status: document.getElementById('modelStatus')?.textContent?.trim(),
    runButton: document.getElementById('runButton')?.textContent?.trim(),
    totalCount: document.getElementById('totalCount')?.textContent?.trim(),
    autoRunDone: typeof state !== 'undefined' ? state.autoRunDone : false,
    imageWidth: typeof state !== 'undefined' && state.image ? state.image.width : 0,
    imageHeight: typeof state !== 'undefined' && state.image ? state.image.height : 0,
    detections: typeof state !== 'undefined' ? state.detections : null,
    debug: typeof state !== 'undefined' ? state.debug : null
  }))()`;
  const started = Date.now();
  let value = null;
  while (Date.now() - started < timeoutMs) {
    const result = await send("Runtime.evaluate", { expression, returnByValue: true });
    if (result.result.exceptionDetails) {
      throw new Error(result.result.exceptionDetails.text || "CDP evaluation failed");
    }
    value = JSON.parse(result.result.result.value);
    if (value.runButton === "Run" && (value.autoRunDone || value.totalCount !== "0")) return value;
    if (value.status === "Model load failed") return value;
    await sleep(1000);
  }
  throw new Error(`Timed out waiting ${timeoutMs}ms for browser autorun`);
}

async function captureScreenshot(send, outputPath) {
  if (!outputPath) return;
  const result = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  const absolute = path.resolve(ROOT, outputPath);
  fs.mkdirSync(path.dirname(absolute), { recursive: true });
  fs.writeFileSync(absolute, Buffer.from(result.result.data, "base64"));
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function writeDetectionCsv(outputPath, detections) {
  if (!outputPath) return;
  const absolute = path.resolve(ROOT, outputPath);
  fs.mkdirSync(path.dirname(absolute), { recursive: true });
  const fieldnames = [
    "index",
    "x1",
    "y1",
    "x2",
    "y2",
    "detector_class",
    "detector_conf",
    "fragment_class",
    "fragment_conf",
    "final_class",
    "final_score",
  ];
  const lines = [fieldnames.join(",")];
  detections.forEach((detection, index) => {
    const row = {
      index,
      x1: detection.x1,
      y1: detection.y1,
      x2: detection.x2,
      y2: detection.y2,
      detector_class: detection.detectorName,
      detector_conf: detection.detectorScore,
      fragment_class: detection.fragmentName,
      fragment_conf: detection.fragmentScore,
      final_class: detection.name,
      final_score: detection.score,
    };
    lines.push(fieldnames.map((field) => csvCell(row[field])).join(","));
  });
  fs.writeFileSync(absolute, `${lines.join("\n")}\n`);
}

function writeSummaryJson(outputPath, summary) {
  if (!outputPath) return;
  const absolute = path.resolve(ROOT, outputPath);
  fs.mkdirSync(path.dirname(absolute), { recursive: true });
  fs.writeFileSync(absolute, `${JSON.stringify(summary, null, 2)}\n`);
}

function browserUrl(args) {
  const params = new URLSearchParams();
  params.set("image", args.image);
  params.set("autorun", "1");
  if (args.proposalConf) params.set("proposalConf", args.proposalConf);
  if (args.detectorOverride) params.set("detectorOverride", args.detectorOverride);
  if (args.detectorModel) params.set("detectorModel", args.detectorModel);
  if (args.fragmentClassifierModel) params.set("fragmentClassifierModel", args.fragmentClassifierModel);
  if (args.stackConfig) params.set("stackConfig", args.stackConfig);
  if (args.rejectFragmentDisagreement) params.set("rejectFragmentDisagreement", "1");
  if (args.fragmentDisagreementMinConf) params.set("fragmentDisagreementMinConf", args.fragmentDisagreementMinConf);
  if (args.unclassifiedMinConf) params.set("unclassifiedMinConf", args.unclassifiedMinConf);
  if (args.nmsIou) params.set("nmsIou", args.nmsIou);
  if (args.cropPadding) params.set("cropPadding", args.cropPadding);
  return `http://127.0.0.1:${args.port}/demo/browser/?${params.toString()}`;
}

async function main() {
  const args = parseArgs(process.argv);
  if (!fs.existsSync(args.edge)) {
    throw new Error(`Edge executable not found: ${args.edge}`);
  }
  const profileDir = path.join(ROOT, ".cache_runtime", "edge-cdp-smoke-profile");
  fs.rmSync(profileDir, { recursive: true, force: true });
  fs.mkdirSync(profileDir, { recursive: true });

  const server = spawn("python", ["-m", "http.server", String(args.port), "--bind", "127.0.0.1"], {
    cwd: ROOT,
    stdio: "ignore",
    windowsHide: true,
  });
  let edge = null;
  try {
    await waitForHttp(`http://127.0.0.1:${args.port}/demo/browser/index.html`, 15000);
    const url = browserUrl(args);
    edge = spawn(
      args.edge,
      [
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        `--remote-debugging-port=${args.debugPort}`,
        `--user-data-dir=${profileDir}`,
        "--window-size=1280,900",
        url,
      ],
      { stdio: "ignore", windowsHide: true },
    );
    const ws = await connectCdp(args.debugPort, 15000);
    const send = createCdpClient(ws);
    const value = await readBrowserState(send, args.timeoutMs);
    await captureScreenshot(send, args.screenshot);
    writeDetectionCsv(args.outCsv, value.detections || []);
    ws.close();
    const summary = summarize(value, args);
    writeSummaryJson(args.outJson, summary);
    console.log(JSON.stringify(summary, null, 2));
    enforceCountContract(summary);
    enforceEvaluation(summary, args);
  } finally {
    killTree(edge);
    killTree(server);
    stopWindowsProcessesByCommandLine(profileDir);
    stopWindowsProcessesByCommandLine(`http.server ${args.port}`);
  }
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
