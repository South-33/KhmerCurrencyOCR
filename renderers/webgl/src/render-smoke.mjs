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

const occluders = [
  {
    kind: "finger_capsule",
    layer: 10,
    color: 0xc58663,
    position: [-0.08, -0.06, 0.19],
    rotation: [0.18, -0.10, 1.36],
    radius: 0.045,
    length: 0.54,
  },
  {
    kind: "finger_capsule",
    layer: 11,
    color: 0xb87958,
    position: [0.32, 0.08, 0.21],
    rotation: [0.12, 0.16, 1.05],
    radius: 0.043,
    length: 0.46,
  },
];

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
const occluders = ${JSON.stringify(occluders)};
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9b927d);

const camera = new THREE.PerspectiveCamera(48, 960 / 720, 0.01, 20);
camera.position.set(0.0, -0.35, 2.2);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
renderer.setSize(960, 720);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
document.body.appendChild(renderer.domElement);

const hemi = new THREE.HemisphereLight(0xf5f1e7, 0x6a5d50, 1.6);
scene.add(hemi);
const key = new THREE.DirectionalLight(0xffe0aa, 2.0);
key.position.set(-1.5, -1.0, 3.0);
key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
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
const backingMeshes = [];
const occluderMeshes = [];

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
    const texture = await loader.loadAsync(asset.textureUrl);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 2;
    const geometry = new THREE.PlaneGeometry(1.28, 0.56, 36, 16);
    bendGeometry(geometry, 0.075, 0.012);
    const backingMaterial = new THREE.MeshStandardMaterial({
      color: 0xf2ead7,
      roughness: 0.92,
      metalness: 0.0,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false
    });
    const backing = new THREE.Mesh(geometry.clone(), backingMaterial);
    backing.position.set(...asset.position);
    backing.rotation.set(...asset.rotation);
    backing.renderOrder = 10 + asset.layer * 3;
    backing.receiveShadow = true;
    backing.userData = { material: backingMaterial };
    backingMeshes.push(backing);
    scene.add(backing);

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
    mesh.renderOrder = 11 + asset.layer * 3;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.userData = { material, idColor: asset.idColor, asset };
    meshes.push(mesh);
    scene.add(mesh);
  }
}

function addOccluders() {
  for (const occluder of occluders) {
    const geometry = new THREE.CapsuleGeometry(occluder.radius, occluder.length, 8, 18);
    const material = new THREE.MeshStandardMaterial({
      color: occluder.color,
      roughness: 0.88,
      metalness: 0.0,
      depthTest: false,
      depthWrite: false
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(...occluder.position);
    mesh.rotation.set(...occluder.rotation);
    mesh.renderOrder = 100 + occluder.layer;
    mesh.castShadow = true;
    mesh.receiveShadow = false;
    mesh.userData = { material, occluder, finalColor: [0, 0, 0], auditColor: [255, 255, 255] };
    occluderMeshes.push(mesh);
    scene.add(mesh);
  }
}

await addNotes();
addOccluders();

window.renderPass = (mode) => {
  if (mode === "id") {
    scene.background = new THREE.Color(0x000000);
    table.visible = false;
    hemi.visible = false;
    key.visible = false;
    for (const mesh of backingMeshes) mesh.visible = false;
    for (const mesh of meshes) {
      const [r, g, b] = mesh.userData.idColor;
      mesh.material = new THREE.MeshBasicMaterial({
        color: new THREE.Color(r / 255, g / 255, b / 255),
        side: THREE.DoubleSide,
        depthTest: false,
        depthWrite: false
      });
    }
    for (const mesh of occluderMeshes) {
      mesh.material = new THREE.MeshBasicMaterial({
        color: new THREE.Color(0x000000),
        depthTest: false,
        depthWrite: false
      });
    }
  } else {
    scene.background = new THREE.Color(0x9b927d);
    table.visible = true;
    hemi.visible = true;
    key.visible = true;
    for (const mesh of backingMeshes) {
      mesh.visible = true;
      mesh.material = mesh.userData.material;
    }
    for (const mesh of meshes) mesh.material = mesh.userData.material;
    for (const mesh of occluderMeshes) mesh.material = mesh.userData.material;
  }
  renderer.render(scene, camera);
};

function captureCanvasPixels() {
  const canvas = renderer.domElement;
  const scratch = document.createElement("canvas");
  scratch.width = canvas.width;
  scratch.height = canvas.height;
  const context = scratch.getContext("2d", { willReadFrequently: true });
  context.drawImage(canvas, 0, 0);
  return context.getImageData(0, 0, scratch.width, scratch.height);
}

function pixelKey(data, offset) {
  return data[offset] + "," + data[offset + 1] + "," + data[offset + 2];
}

window.extractIdBoxes = () => {
  for (const mesh of meshes) mesh.visible = true;
  for (const mesh of occluderMeshes) mesh.visible = true;
  window.renderPass("id");
  const { data, width, height } = captureCanvasPixels();
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

window.auditLayerOrder = () => {
  for (const mesh of meshes) mesh.visible = true;
  for (const mesh of occluderMeshes) mesh.visible = true;
  window.renderPass("id");
  const finalPass = captureCanvasPixels();
  const finalData = finalPass.data;
  const width = finalPass.width;
  const height = finalPass.height;

  const noteItems = meshes.map((mesh) => ({
    mesh,
    className: mesh.userData.asset.className,
    layer: mesh.userData.asset.layer,
    auditColor: mesh.userData.idColor,
    finalColorKey: mesh.userData.idColor.join(","),
    isOccluder: false,
  }));
  const occluderItems = occluderMeshes.map((mesh, index) => ({
    mesh,
    className: "occluder_" + index,
    layer: mesh.userData.occluder.layer,
    auditColor: mesh.userData.auditColor,
    finalColorKey: mesh.userData.finalColor.join(","),
    isOccluder: true,
  }));

  const isolated = [...noteItems, ...occluderItems].map((item) => {
    for (const mesh of meshes) mesh.visible = mesh === item.mesh;
    for (const mesh of occluderMeshes) mesh.visible = mesh === item.mesh;
    scene.background = new THREE.Color(0x000000);
    table.visible = false;
    hemi.visible = false;
    key.visible = false;
    const [r, g, b] = item.auditColor;
    item.mesh.material = new THREE.MeshBasicMaterial({
      color: new THREE.Color(r / 255, g / 255, b / 255),
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide
    });
    renderer.render(scene, camera);
    return {
      className: item.className,
      layer: item.layer,
      auditColorKey: item.auditColor.join(","),
      finalColorKey: item.finalColorKey,
      isOccluder: item.isOccluder,
      data: new Uint8ClampedArray(captureCanvasPixels().data),
    };
  });

  for (const mesh of meshes) mesh.visible = true;
  for (const mesh of occluderMeshes) mesh.visible = true;
  window.renderPass("id");

  let visiblePixels = 0;
  let overlapPixels = 0;
  let occluderPixels = 0;
  let violations = 0;
  const examples = [];

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      let coverage = 0;
      let expected = null;
      for (const item of isolated) {
        if (pixelKey(item.data, offset) === "0,0,0") continue;
        coverage += 1;
        if (expected === null || item.layer > expected.layer) expected = item;
      }
      if (coverage === 0) continue;
      visiblePixels += 1;
      if (coverage > 1) overlapPixels += 1;
      if (expected.isOccluder) occluderPixels += 1;
      const actualKey = pixelKey(finalData, offset);
      if (actualKey !== expected.finalColorKey) {
        violations += 1;
        if (examples.length < 10) {
          examples.push({ x, y, expected: expected.className, expectedColor: expected.finalColorKey, actualColor: actualKey, coverage });
        }
      }
    }
  }

  return { width, height, visiblePixels, overlapPixels, occluderPixels, violations, examples };
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
  const textureAssets = assets.map((asset) => ({ ...asset, textureUrl: pathToFileURL(asset.path).href }));
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
    const layerAudit = await page.evaluate(() => window.auditLayerOrder());
    if (layerAudit.violations !== 0) {
      throw new Error(`Layer-order audit failed with ${layerAudit.violations} violating pixels`);
    }
    fs.writeFileSync(path.join(OUT_DIR, "visible_boxes.json"), JSON.stringify({ boxes }, null, 2));
    fs.writeFileSync(path.join(OUT_DIR, "layer_audit.json"), JSON.stringify(layerAudit, null, 2));
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
        assets: textureAssets,
        occluders,
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
