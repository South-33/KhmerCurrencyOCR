const CLASSES = [
  "",
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

const state = {
  rows: [],
  headers: [],
};

const manifestPath = document.getElementById("manifestPath");
const loadButton = document.getElementById("loadButton");
const exportButton = document.getElementById("exportButton");
const includedOnly = document.getElementById("includedOnly");
const summary = document.getElementById("summary");
const grid = document.getElementById("grid");

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (quoted) {
      if (char === '"' && next === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (char !== "\r") {
      field += char;
    }
  }
  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }
  const headers = rows.shift() || [];
  return {
    headers,
    rows: rows.filter((item) => item.length > 1).map((item) => Object.fromEntries(headers.map((key, index) => [key, item[index] || ""]))),
  };
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(headers, rows) {
  return [headers.join(","), ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(","))].join("\n");
}

function repoUrl(path) {
  return `/${path.replaceAll("\\", "/").replace(/^\/+/, "")}`;
}

function predictedClass(row) {
  return row.review_class || row.fragment_class || row.detector_class || "";
}

function rowMeta(row) {
  const title = row.crop_id || row.proposal_index || row.source_image || "";
  const detail = [
    row.failure_pair || "",
    row.class_name || row.fragment_class || row.detector_class || "",
    row.side || "",
    row.confidence ? `conf ${row.confidence}` : "",
    row.box_area_frac ? `area ${row.box_area_frac}` : "",
    row.fragment_conf || row.detector_conf || "",
  ].filter(Boolean).join(" · ");
  return { title, detail };
}

function render() {
  const rows = includedOnly.checked ? state.rows.filter((row) => row.review_include) : state.rows;
  const included = state.rows.filter((row) => row.review_include).length;
  summary.textContent = `${state.rows.length} rows, ${included} included`;
  grid.innerHTML = "";
  for (const row of rows) {
    const card = document.createElement("article");
    card.className = "card";
    const classOptions = CLASSES.map((name) => `<option value="${name}" ${name === predictedClass(row) ? "selected" : ""}>${name || "skip"}</option>`).join("");
    const meta = rowMeta(row);
    card.innerHTML = `
      <img src="${repoUrl(row.crop_path)}" alt="" />
      <div class="body">
        <div class="meta">
          <span>${meta.title}</span>
          <span>${meta.detail}</span>
        </div>
        <label class="row">
          <input class="include" type="checkbox" ${row.review_include ? "checked" : ""} />
          <span>Include</span>
        </label>
        <select class="class-select">${classOptions}</select>
        <textarea class="notes" placeholder="notes">${row.review_notes || ""}</textarea>
      </div>
    `;
    card.querySelector(".include").addEventListener("change", (event) => {
      row.review_include = event.target.checked ? "1" : "";
      render();
    });
    card.querySelector(".class-select").addEventListener("change", (event) => {
      row.review_class = event.target.value;
    });
    card.querySelector(".notes").addEventListener("input", (event) => {
      row.review_notes = event.target.value;
    });
    grid.append(card);
  }
}

async function loadManifest() {
  loadButton.disabled = true;
  try {
    const response = await fetch(manifestPath.value);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const parsed = parseCsv(await response.text());
    state.headers = parsed.headers;
    for (const required of ["review_include", "review_class", "review_notes"]) {
      if (!state.headers.includes(required)) {
        state.headers.push(required);
      }
    }
    state.rows = parsed.rows;
    exportButton.disabled = false;
    render();
  } catch (error) {
    summary.textContent = `Load failed: ${error.message}`;
  } finally {
    loadButton.disabled = false;
  }
}

function exportCsv() {
  const blob = new Blob([toCsv(state.headers, state.rows)], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "review.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

loadButton.addEventListener("click", loadManifest);
exportButton.addEventListener("click", exportCsv);
includedOnly.addEventListener("change", render);

const params = new URLSearchParams(window.location.search);
const initialManifest = params.get("manifest");
if (initialManifest) {
  manifestPath.value = initialManifest;
  loadManifest();
}
