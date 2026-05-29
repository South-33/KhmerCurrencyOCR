const CLASSES = [
  "",
  "banknote_unknown",
  "background",
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

const PRESET_MANIFESTS = [
  {
    label: "Real fan proposal review",
    path: "/data/review/real_fan_candidate_proposal_review_v1/review.csv",
  },
  {
    label: "Roboflow old/common failures",
    path: "/data/review/roboflow_cuurecy_detection_is_oldcommon_highconf_failure_review_v1/manifest.csv",
  },
  {
    label: "P1 old/common focus queue",
    path: "/data/review/cashsnap_p1_oldcommon_partial_focus_review_v1/manifest.csv",
  },
  {
    label: "P1 old/common failure queue",
    path: "/data/review/p1_focus_v2_oldcommon_failure_review_v1/manifest.csv",
  },
  {
    label: "Roboflow 5k/10k partials",
    path: "/data/review/roboflow_cuurecy_detection_is_khr_5k_10k_partial_review_v1/manifest.csv",
  },
  {
    label: "Roboflow 20k/50k partials",
    path: "/data/review/roboflow_cuurecy_detection_is_khr_20k_50k_partial_review_v1/manifest.csv",
  },
  {
    label: "CashSnap old/common crops",
    path: "/data/review/cashsnap_old_common_khr_crop_review_v1/manifest.csv",
  },
];

const state = {
  rows: [],
  headers: [],
};

const presetManifest = document.getElementById("presetManifest");
const manifestPath = document.getElementById("manifestPath");
const loadButton = document.getElementById("loadButton");
const exportButton = document.getElementById("exportButton");
const clearDraftButton = document.getElementById("clearDraftButton");
const pairFilter = document.getElementById("pairFilter");
const sideFilter = document.getElementById("sideFilter");
const includedOnly = document.getElementById("includedOnly");
const draftStatus = document.getElementById("draftStatus");
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

function htmlEscape(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
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

function rowKey(row) {
  return row.crop_id || row.crop_path || row.source_crop || row.source_image || "";
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
  ].filter(Boolean).join(" | ");
  return { title, detail };
}

function rowLinks(row) {
  const links = [{ label: "Crop", href: repoUrl(row.crop_path) }];
  if (row.source_image) {
    links.push({ label: "Source", href: repoUrl(row.source_image) });
  }
  return links
    .filter((link) => link.href)
    .map((link) => `<a href="${htmlEscape(link.href)}" target="_blank" rel="noreferrer">${htmlEscape(link.label)}</a>`)
    .join("");
}

function uniqueValues(key) {
  return [...new Set(state.rows.map((row) => row[key]).filter(Boolean))].sort();
}

function setOptions(select, label, values) {
  select.innerHTML = [
    `<option value="">${label}</option>`,
    ...values.map((value) => `<option value="${htmlEscape(value)}">${htmlEscape(value)}</option>`),
  ].join("");
  select.disabled = values.length === 0;
}

function refreshFilters() {
  setOptions(pairFilter, "All pairs", uniqueValues("failure_pair"));
  setOptions(sideFilter, "All sides", uniqueValues("side"));
}

function refreshPresetSelection() {
  const match = PRESET_MANIFESTS.find((preset) => preset.path === manifestPath.value);
  presetManifest.value = match ? match.path : "";
}

function draftKey() {
  return `cashsnap-review:${manifestPath.value}`;
}

function loadDraftRows() {
  const raw = localStorage.getItem(draftKey());
  if (!raw) {
    return new Map();
  }
  try {
    const parsed = JSON.parse(raw);
    return new Map(parsed.rows.map((row) => [row.key, row]));
  } catch {
    return new Map();
  }
}

function applyDraft() {
  const draftRows = loadDraftRows();
  if (!draftRows.size) {
    draftStatus.textContent = "No draft";
    clearDraftButton.disabled = true;
    return;
  }
  let restored = 0;
  for (const row of state.rows) {
    const saved = draftRows.get(rowKey(row));
    if (!saved) {
      continue;
    }
    row.review_include = saved.review_include || "";
    row.review_class = saved.review_class || "";
    row.review_notes = saved.review_notes || "";
    restored += 1;
  }
  draftStatus.textContent = `Restored ${restored} draft rows`;
  clearDraftButton.disabled = false;
}

function saveDraft() {
  const rows = state.rows
    .map((row) => ({
      key: rowKey(row),
      review_include: row.review_include || "",
      review_class: row.review_class || "",
      review_notes: row.review_notes || "",
    }))
    .filter((row) => row.key && (row.review_include || row.review_class || row.review_notes));
  if (!rows.length) {
    localStorage.removeItem(draftKey());
    draftStatus.textContent = "No draft";
    clearDraftButton.disabled = true;
    return;
  }
  localStorage.setItem(draftKey(), JSON.stringify({ saved_at: new Date().toISOString(), rows }));
  draftStatus.textContent = `Draft saved for ${rows.length} rows`;
  clearDraftButton.disabled = false;
}

function visibleRows() {
  return state.rows.filter((row) => {
    if (includedOnly.checked && !row.review_include) {
      return false;
    }
    if (pairFilter.value && row.failure_pair !== pairFilter.value) {
      return false;
    }
    if (sideFilter.value && row.side !== sideFilter.value) {
      return false;
    }
    return true;
  });
}

function render() {
  const rows = visibleRows();
  const included = state.rows.filter((row) => row.review_include).length;
  summary.textContent = `${rows.length}/${state.rows.length} rows visible, ${included} included`;
  grid.innerHTML = "";
  for (const row of rows) {
    const card = document.createElement("article");
    card.className = "card";
    const classOptions = CLASSES.map((name) => `<option value="${name}" ${name === predictedClass(row) ? "selected" : ""}>${name || "skip"}</option>`).join("");
    const meta = rowMeta(row);
    const links = rowLinks(row);
    card.innerHTML = `
      <img src="${repoUrl(row.crop_path)}" alt="" />
      <div class="body">
        <div class="meta">
          <span>${meta.title}</span>
          <span>${meta.detail}</span>
        </div>
        <div class="links">${links}</div>
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
      saveDraft();
      render();
    });
    card.querySelector(".class-select").addEventListener("change", (event) => {
      row.review_class = event.target.value;
      saveDraft();
    });
    card.querySelector(".notes").addEventListener("input", (event) => {
      row.review_notes = event.target.value;
      saveDraft();
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
    applyDraft();
    refreshFilters();
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

function clearDraft() {
  localStorage.removeItem(draftKey());
  draftStatus.textContent = "No draft";
  clearDraftButton.disabled = true;
}

loadButton.addEventListener("click", loadManifest);
exportButton.addEventListener("click", exportCsv);
clearDraftButton.addEventListener("click", clearDraft);
includedOnly.addEventListener("change", render);
pairFilter.addEventListener("change", render);
sideFilter.addEventListener("change", render);
manifestPath.addEventListener("input", refreshPresetSelection);
presetManifest.addEventListener("change", () => {
  if (presetManifest.value) {
    manifestPath.value = presetManifest.value;
    loadManifest();
  }
});

presetManifest.innerHTML = [
  `<option value="">Custom manifest</option>`,
  ...PRESET_MANIFESTS.map((preset) => `<option value="${htmlEscape(preset.path)}">${htmlEscape(preset.label)}</option>`),
].join("");

const params = new URLSearchParams(window.location.search);
const initialManifest = params.get("manifest");
if (initialManifest) {
  manifestPath.value = initialManifest;
  refreshPresetSelection();
  loadManifest();
} else {
  refreshPresetSelection();
}
