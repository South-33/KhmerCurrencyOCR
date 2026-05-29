const fs = require("node:fs");
const path = require("node:path");
const { spawn, spawnSync } = require("node:child_process");

const ROOT = path.resolve(__dirname, "..");
const DEFAULT_EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";

function parseArgs(argv) {
  const args = {
    manifest: "/data/review/p1_focus_v2_oldcommon_failure_review_v1/manifest.csv",
    port: 8787,
    debugPort: 9231,
    timeoutMs: 30000,
    edge: process.env.EDGE_PATH || DEFAULT_EDGE,
  };
  for (let index = 2; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (key === "--manifest") {
      args.manifest = value;
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
    } else if (key === "--edge") {
      args.edge = value;
      index += 1;
    } else {
      throw new Error(`Unknown argument: ${key}`);
    }
  }
  return args;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function killTree(child) {
  if (!child?.pid) return;
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
      const page = pages.find((item) => (item.url || "").includes("/demo/review/")) || pages[0];
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

async function evaluate(send, expression) {
  const result = await send("Runtime.evaluate", { expression, returnByValue: true });
  if (result.result.exceptionDetails) {
    throw new Error(result.result.exceptionDetails.text || "CDP evaluation failed");
  }
  return result.result.result.value;
}

async function waitForRows(send, timeoutMs) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const value = await evaluate(
      send,
      `(() => JSON.stringify({
        rows: typeof state !== 'undefined' ? state.rows.length : 0,
        cards: document.querySelectorAll('.card').length,
        summary: document.getElementById('summary')?.textContent || '',
        draft: document.getElementById('draftStatus')?.textContent || ''
      }))()`,
    );
    const parsed = JSON.parse(value);
    if (parsed.rows > 0 && parsed.cards > 0) return parsed;
    await sleep(250);
  }
  throw new Error("Timed out waiting for review rows");
}

async function main() {
  const args = parseArgs(process.argv);
  if (!fs.existsSync(args.edge)) {
    throw new Error(`Edge executable not found: ${args.edge}`);
  }
  const profileDir = path.join(ROOT, ".cache_runtime", "edge-review-smoke-profile");
  fs.rmSync(profileDir, { recursive: true, force: true });
  fs.mkdirSync(profileDir, { recursive: true });

  const server = spawn("python", ["-m", "http.server", String(args.port), "--bind", "127.0.0.1"], {
    cwd: ROOT,
    stdio: "ignore",
    windowsHide: true,
  });
  let edge = null;
  try {
    await waitForHttp(`http://127.0.0.1:${args.port}/demo/review/index.html`, 15000);
    const url = `http://127.0.0.1:${args.port}/demo/review/?manifest=${encodeURIComponent(args.manifest)}`;
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
    await send("Runtime.enable");
    await send("Page.enable");
    const loaded = await waitForRows(send, args.timeoutMs);
    await evaluate(
      send,
      `(() => {
        document.querySelector('.include').click();
        const select = document.querySelector('.class-select');
        select.value = 'KHR_5000';
        select.dispatchEvent(new Event('change', { bubbles: true }));
        const notes = document.querySelector('.notes');
        notes.value = 'smoke draft';
        notes.dispatchEvent(new Event('input', { bubbles: true }));
      })()`,
    );
    const saved = JSON.parse(
      await evaluate(
        send,
        `(() => JSON.stringify({
          draft: document.getElementById('draftStatus').textContent,
          stored: localStorage.getItem('cashsnap-review:${args.manifest}') !== null
        }))()`,
      ),
    );
    if (!saved.stored || !saved.draft.includes("Draft saved")) {
      throw new Error(`Draft was not saved: ${JSON.stringify(saved)}`);
    }
    await send("Page.reload", { ignoreCache: true });
    const restored = await waitForRows(send, args.timeoutMs);
    const restoredValues = JSON.parse(
      await evaluate(
        send,
        `(() => JSON.stringify({
          draft: document.getElementById('draftStatus').textContent,
          checked: document.querySelector('.include').checked,
          klass: document.querySelector('.class-select').value,
          notes: document.querySelector('.notes').value
        }))()`,
      ),
    );
    if (!restoredValues.draft.includes("Restored") || !restoredValues.checked || restoredValues.klass !== "KHR_5000" || restoredValues.notes !== "smoke draft") {
      throw new Error(`Draft was not restored: ${JSON.stringify(restoredValues)}`);
    }
    ws.close();
    console.log(JSON.stringify({ status: "ok", loaded, saved, restored, restoredValues }, null, 2));
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
