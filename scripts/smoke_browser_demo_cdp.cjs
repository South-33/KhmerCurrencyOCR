const fs = require("node:fs");
const path = require("node:path");
const { spawn, spawnSync } = require("node:child_process");

const ROOT = path.resolve(__dirname, "..");
const DEFAULT_EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
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

function parseArgs(argv) {
  const args = {
    image: "/data/real_fan_benchmark/images/candidates/real_overlap_0003_commons_shop_5k_10k_20k.png",
    port: 8787,
    debugPort: 9223,
    timeoutMs: 120000,
    screenshot: "",
    outCsv: "",
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
  --screenshot PATH   Optional PNG screenshot output path.
  --out-csv PATH      Optional CSV output for browser-side detections.
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

function summarize(value) {
  const counts = {};
  let khrValue = 0;
  let usdValue = 0;
  for (const detection of value.detections || []) {
    counts[detection.name] = (counts[detection.name] || 0) + 1;
    if (detection.name.startsWith("KHR_")) {
      khrValue += VALUES[detection.name] || 0;
    } else if (detection.name.startsWith("USD_")) {
      usdValue += VALUES[detection.name] || 0;
    }
  }
  return {
    status: value.status,
    totalCount: Number(value.totalCount || 0),
    khrValue,
    usdValue,
    predClasses: Object.fromEntries(Object.entries(counts).sort(([left], [right]) => left.localeCompare(right))),
    detections: value.detections || [],
  };
}

async function readBrowserState(send, timeoutMs) {
  await send("Runtime.enable");
  const expression = `(() => JSON.stringify({
    status: document.getElementById('modelStatus')?.textContent?.trim(),
    runButton: document.getElementById('runButton')?.textContent?.trim(),
    totalCount: document.getElementById('totalCount')?.textContent?.trim(),
    detections: typeof state !== 'undefined' ? state.detections : null
  }))()`;
  const started = Date.now();
  let value = null;
  while (Date.now() - started < timeoutMs) {
    const result = await send("Runtime.evaluate", { expression, returnByValue: true });
    if (result.result.exceptionDetails) {
      throw new Error(result.result.exceptionDetails.text || "CDP evaluation failed");
    }
    value = JSON.parse(result.result.result.value);
    if (value.totalCount !== "0" && value.runButton === "Run") return value;
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
    const url = `http://127.0.0.1:${args.port}/demo/browser/?image=${args.image}&autorun=1`;
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
    console.log(JSON.stringify(summarize(value), null, 2));
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
