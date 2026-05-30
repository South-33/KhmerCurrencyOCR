import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import puppeteer from "puppeteer-core";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..", "..", "..");
const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const OUT_DIR = path.join(ROOT, "data", "synthetic", "cashsnap_webgl_smoke");
const THREE_MODULE = pathToFileURL(path.join(ROOT, "renderers", "webgl", "node_modules", "three", "build", "three.module.js")).href;

const assets = [
  {
    className: "KHR_5000",
    idColor: [255, 0, 0],
    path: path.join(ROOT, "data", "asset_candidates", "numista_current_cutout_bank_v1", "KHR_5000", "KHR_5000_2015_front.png"),
    position: [-0.34, 0.02, 0.03],
    rotation: [0.08, -0.12, -0.18],
    layer: 0,
  },
  {
    className: "KHR_10000",
    idColor: [0, 255, 0],
    path: path.join(ROOT, "data", "asset_candidates", "numista_current_cutout_bank_v1", "KHR_10000", "KHR_10000_2015_front.png"),
    position: [0.10, -0.04, 0.13],
    rotation: [-0.03, 0.16, 0.16],
    layer: 2,
  },
  {
    className: "KHR_20000",
    idColor: [0, 0, 255],
    path: path.join(ROOT, "data", "asset_candidates", "numista_current_cutout_bank_v1", "KHR_20000", "KHR_20000_2017_front.png"),
    position: [0.00, 0.24, 0.08],
    rotation: [0.04, -0.05, 0.03],
    layer: 1,
  },
];

function imageDataUrl(filePath) {
  const bytes = fs.readFileSync(filePath);
  return `data:image/png;base64,${bytes.toString("base64")}`;
}

function html(textureAssets) {
  return `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #918b78; }
    canvas { display: block; }
  </style>
  <script type="importmap">
    {"imports":{"three":"${THREE_MODULE}"}}
  </script>
</head>
<body>
<script type="module">
import * as THREE from "three";

const assets = ${JSON.stringify(textureAssets)};
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9b927d);

const camera = new THREE.PerspectiveCamera(48, 960 / 720, 0.01, 20);
camera.position.set(0.0, -0.35, 2.2);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
renderer.setSize(960, 720);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const hemi = new THREE.HemisphereLight(0xf5f1e7, 0x6a5d50, 1.6);
scene.add(hemi);
const key = new THREE.DirectionalLight(0xffe0aa, 2.0);
key.position.set(-1.5, -1.0, 3.0);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
scene.add(key);

const table = new THREE.Mesh(
  new THREE.PlaneGeometry(4.0, 3.0, 4, 4),
  new THREE.MeshStandardMaterial({ color: 0x9d8460, roughness: 0.86 })
);
table.receiveShadow = true;
table.position.z = -0.02;
scene.add(table);

const loader = new THREE.TextureLoader();
const meshes = [];

function bendGeometry(geometry, curl, ripple) {
  const pos = geometry.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = curl * x * x + ripple * Math.sin(x * 10.0) * Math.sin((y + 0.34) * 7.0);
    pos.setZ(i, z);
  }
  pos.needsUpdate = true;
  geometry.computeVertexNormals();
}

async function addNotes() {
  for (const asset of [...assets].sort((a, b) => a.layer - b.layer)) {
    const texture = await loader.loadAsync(asset.dataUrl);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 4;
    const geometry = new THREE.PlaneGeometry(1.28, 0.56, 36, 16);
    bendGeometry(geometry, 0.075, 0.012);
    const material = new THREE.MeshStandardMaterial({
      map: texture,
      roughness: 0.74,
      metalness: 0.0,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(...asset.position);
    mesh.rotation.set(...asset.rotation);
    mesh.renderOrder = 10 + asset.layer;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.userData = { material, idColor: asset.idColor };
    meshes.push(mesh);
    scene.add(mesh);
  }
}

await addNotes();

window.renderPass = (mode) => {
  if (mode === "id") {
    scene.background = new THREE.Color(0x000000);
    table.visible = false;
    hemi.visible = false;
    key.visible = false;
    for (const mesh of meshes) {
      const [r, g, b] = mesh.userData.idColor;
      mesh.material = new THREE.MeshBasicMaterial({
        color: new THREE.Color(r / 255, g / 255, b / 255),
        side: THREE.DoubleSide,
        depthTest: false,
        depthWrite: false
      });
    }
  } else {
    scene.background = new THREE.Color(0x9b927d);
    table.visible = true;
    hemi.visible = true;
    key.visible = true;
    for (const mesh of meshes) mesh.material = mesh.userData.material;
  }
  renderer.render(scene, camera);
};

window.extractIdBoxes = () => {
  window.renderPass("id");
  const canvas = renderer.domElement;
  const scratch = document.createElement("canvas");
  scratch.width = canvas.width;
  scratch.height = canvas.height;
  const context = scratch.getContext("2d", { willReadFrequently: true });
  context.drawImage(canvas, 0, 0);
  const { data, width, height } = context.getImageData(0, 0, scratch.width, scratch.height);
  const boxes = new Map();

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      if (r === 0 && g === 0 && b === 0) continue;
      const key = r + "," + g + "," + b;
      const box = boxes.get(key) ?? { color: [r, g, b], minX: x, minY: y, maxX: x, maxY: y, pixels: 0 };
      box.minX = Math.min(box.minX, x);
      box.minY = Math.min(box.minY, y);
      box.maxX = Math.max(box.maxX, x);
      box.maxY = Math.max(box.maxY, y);
      box.pixels += 1;
      boxes.set(key, box);
    }
  }

  const colorToClass = new Map(assets.map((asset, index) => [asset.idColor.join(","), { classIndex: index, className: asset.className }]));
  return [...boxes.entries()].map(([key, box]) => ({
    ...colorToClass.get(key),
    ...box,
    width,
    height,
    yolo: [
      ((box.minX + box.maxX + 1) / 2) / width,
      ((box.minY + box.maxY + 1) / 2) / height,
      (box.maxX - box.minX + 1) / width,
      (box.maxY - box.minY + 1) / height,
    ],
  })).sort((a, b) => a.classIndex - b.classIndex);
};

window.renderPass("visual");
window.__cashsnapReady = true;
</script>
</body>
</html>`;
}

async function main() {
  if (!fs.existsSync(EDGE)) {
    throw new Error(`Microsoft Edge executable not found at ${EDGE}`);
  }
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const textureAssets = assets.map((asset) => ({ ...asset, dataUrl: imageDataUrl(asset.path) }));
  const browser = await puppeteer.launch({
    executablePath: EDGE,
    headless: "new",
    args: [
      "--allow-file-access-from-files",
      "--disable-background-timer-throttling",
      "--disable-renderer-backgrounding",
    ],
  });
  try {
    const page = await browser.newPage();
    page.on("console", (message) => console.log(`[browser:${message.type()}] ${message.text()}`));
    page.on("pageerror", (error) => console.error(`[browser:pageerror] ${error.message}`));
    page.setDefaultTimeout(60000);
    page.setDefaultNavigationTimeout(60000);
    await page.setViewport({ width: 960, height: 720, deviceScaleFactor: 1 });
    const htmlPath = path.join(OUT_DIR, "smoke.html");
    fs.writeFileSync(htmlPath, html(textureAssets));
    await page.goto(pathToFileURL(htmlPath).href, { waitUntil: "domcontentloaded", timeout: 60000 });
    await page.waitForFunction("window.__cashsnapReady === true");
    await page.evaluate(() => window.renderPass("visual"));
    await page.screenshot({ path: path.join(OUT_DIR, "visual.png") });
    await page.evaluate(() => window.renderPass("id"));
    await page.screenshot({ path: path.join(OUT_DIR, "id.png") });
    const boxes = await page.evaluate(() => window.extractIdBoxes());
    fs.writeFileSync(path.join(OUT_DIR, "visible_boxes.json"), JSON.stringify({ boxes }, null, 2));
    fs.writeFileSync(
      path.join(OUT_DIR, "labels_visible.txt"),
      `${boxes.map((box) => `${box.classIndex} ${box.yolo.map((value) => Number(value).toFixed(6)).join(" ")}`).join("\n")}\n`
    );
    fs.writeFileSync(
      path.join(OUT_DIR, "metadata.json"),
      JSON.stringify({
        renderer: "three-webgl-edge",
        visibilityModel: "explicit-layer-order",
        noteDepthPolicy: "banknote planes use renderOrder with depthTest/depthWrite disabled to avoid impossible surface interpenetration in visible masks",
        assets: textureAssets.map(({ dataUrl, ...rest }) => rest),
      }, null, 2)
    );
    console.log(`wrote ${OUT_DIR}`);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
