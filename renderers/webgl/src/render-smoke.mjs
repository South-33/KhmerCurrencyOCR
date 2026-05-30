import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import puppeteer from "puppeteer-core";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..", "..", "..");
const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const THREE_MODULE = pathToFileURL(path.join(ROOT, "renderers", "webgl", "node_modules", "three", "build", "three.module.js")).href;

function argValue(name, defaultValue) {
  const index = process.argv.indexOf(name);
  if (index === -1) return defaultValue;
  const value = process.argv[index + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`Missing value for ${name}`);
  }
  return value;
}

const OUT_DIR = path.resolve(ROOT, argValue("--out-dir", path.join("data", "synthetic", "cashsnap_webgl_smoke")));
const VARIANT = Number.parseInt(argValue("--variant", "0"), 10);
const SCENE_MODE = argValue("--scene-mode", "auto");
const BACKGROUND_DIR = argValue("--background-dir", "");
const WIDTH = Number.parseInt(argValue("--width", "1440"), 10);
const HEIGHT = Number.parseInt(argValue("--height", "1080"), 10);
const VISUAL_SCALE = Number.parseFloat(argValue("--visual-scale", "2"));
const MIN_VISIBLE_PIXELS = Number.parseInt(argValue("--min-visible-pixels", "500"), 10);
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
const ASSET_BANK_DIR = path.join(ROOT, "data", "asset_candidates", "numista_current_cutout_bank_v1");
const PHYSICAL_WIDTH_MM = {
  USD_1: 156,
  USD_5: 156,
  USD_10: 156,
  USD_20: 156,
  USD_50: 156,
  USD_100: 156,
  KHR_500: 138,
  KHR_1000: 142,
  KHR_2000: 146,
  KHR_5000: 142,
  KHR_10000: 155,
  KHR_20000: 155,
  KHR_50000: 155,
};
const INSTANCE_ID_COLORS = [
  [255, 0, 0],
  [0, 255, 0],
  [0, 0, 255],
  [255, 255, 0],
  [255, 0, 255],
  [0, 255, 255],
  [255, 128, 0],
  [128, 0, 255],
];

if (!Number.isInteger(VARIANT) || VARIANT < 0) {
  throw new Error("--variant must be a non-negative integer");
}

if (!Number.isInteger(WIDTH) || !Number.isInteger(HEIGHT) || WIDTH < 320 || HEIGHT < 240) {
  throw new Error("--width/--height must be integer image dimensions >= 320x240");
}

if (!Number.isFinite(VISUAL_SCALE) || VISUAL_SCALE < 1 || VISUAL_SCALE > 4) {
  throw new Error("--visual-scale must be a finite number from 1 to 4");
}

if (!Number.isInteger(MIN_VISIBLE_PIXELS) || MIN_VISIBLE_PIXELS < 1) {
  throw new Error("--min-visible-pixels must be a positive integer");
}

if (!["auto", "clean", "stack", "fan", "qa3"].includes(SCENE_MODE)) {
  throw new Error("--scene-mode must be one of: auto, clean, stack, fan, qa3");
}

const effectiveSceneMode = SCENE_MODE === "auto" ? (VARIANT >= 100 ? "fan" : "stack") : SCENE_MODE;

function mulberry32(seed) {
  return () => {
    let value = seed += 0x6D2B79F5;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}

function randomBetween(rng, min, max) {
  return min + (max - min) * rng();
}

function randomInt(rng, maxExclusive) {
  return Math.floor(rng() * maxExclusive);
}

function rotate2([x, y], angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [x * c - y * s, x * s + y * c];
}

function classIndexFor(variant, index) {
  return (variant * 7 + index * 5) % CLASS_NAMES.length;
}

function hexToNumber(hex) {
  return Number.parseInt(hex.replace("#", ""), 16);
}

function listImageFiles(directory) {
  if (!directory) return [];
  const resolved = path.resolve(ROOT, directory);
  if (!fs.existsSync(resolved)) return [];
  return fs.readdirSync(resolved)
    .filter((name) => /\.(png|jpe?g|webp)$/i.test(name))
    .map((name) => path.join(resolved, name))
    .sort();
}

function sceneConfig(variant, mode, backgroundPath) {
  const modeOffset = mode === "fan" ? 1009 : mode === "clean" ? 2003 : mode === "qa3" ? 3001 : 0;
  const rng = mulberry32(26058003 + variant * 191 + modeOffset);
  const surfaces = [
    { name: "warm_wood", base: "#9b784a", light: "#fff2d6", dark: "#231810", scene: "#9b927d", repeat: [2.5, 2.0] },
    { name: "gray_counter", base: "#827f77", light: "#dedbd2", dark: "#3a3834", scene: "#86837b", repeat: [1.8, 1.8] },
    { name: "green_plastic_mat", base: "#456f5b", light: "#b7d8c5", dark: "#1c3229", scene: "#506f61", repeat: [2.0, 1.7] },
    { name: "blue_shop_table", base: "#566f83", light: "#d6e4ef", dark: "#25323b", scene: "#627487", repeat: [2.1, 1.9] },
    { name: "dark_laminate", base: "#51473c", light: "#b5a28a", dark: "#1f1a15", scene: "#5a5147", repeat: [2.8, 2.2] },
  ];
  const surface = surfaces[randomInt(rng, surfaces.length)];
  return {
    surface: {
      ...surface,
      base: hexToNumber(surface.base),
      light: hexToNumber(surface.light),
      dark: hexToNumber(surface.dark),
      scene: hexToNumber(surface.scene),
      background: backgroundPath ? {
        path: backgroundPath,
        textureUrl: pathToFileURL(backgroundPath).href,
      } : null,
    },
    camera: {
      fov: randomBetween(rng, 42, 56),
      position: [
        randomBetween(rng, -0.10, 0.10),
        randomBetween(rng, -0.48, -0.24),
        randomBetween(rng, 2.05, 2.42),
      ],
      lookAt: [
        randomBetween(rng, -0.05, 0.05),
        randomBetween(rng, -0.03, 0.05),
        0,
      ],
    },
    lighting: {
      hemiIntensity: randomBetween(rng, 1.15, 1.95),
      keyIntensity: randomBetween(rng, 1.35, 2.55),
      keyColor: [0xffd8aa, 0xffffff, 0xdff1ff, 0xffefc8][randomInt(rng, 4)],
      keyPosition: [
        randomBetween(rng, -2.2, 2.2),
        randomBetween(rng, -3.0, -1.2),
        randomBetween(rng, 2.4, 4.4),
      ],
    },
    postprocess: {
      contrast: randomBetween(rng, 1.00, 1.07),
      saturation: randomBetween(rng, 0.90, 1.07),
      brightness: randomBetween(rng, 0.96, 1.05),
      grainAlpha: randomBetween(rng, 14, 30),
      vignette: randomBetween(rng, 42, 78),
    },
  };
}

const baseAssets = [
  {
    className: "KHR_5000",
    classIndex: 9,
    idColor: [255, 0, 0],
    path: path.join(ROOT, "data", "asset_candidates", "numista_current_cutout_bank_v1", "KHR_5000", "KHR_5000_2015_front.png"),
    position: [-0.34, 0.02, 0.03],
    rotation: [0.08, -0.12, -0.18],
    layer: 0,
  },
  {
    className: "KHR_10000",
    classIndex: 10,
    idColor: [0, 255, 0],
    path: path.join(ROOT, "data", "asset_candidates", "numista_current_cutout_bank_v1", "KHR_10000", "KHR_10000_2015_front.png"),
    position: [0.10, -0.04, 0.13],
    rotation: [-0.03, 0.16, 0.16],
    layer: 2,
  },
  {
    className: "KHR_20000",
    classIndex: 11,
    idColor: [0, 0, 255],
    path: path.join(ROOT, "data", "asset_candidates", "numista_current_cutout_bank_v1", "KHR_20000", "KHR_20000_2017_front.png"),
    position: [0.00, 0.24, 0.08],
    rotation: [0.04, -0.05, 0.03],
    layer: 1,
  },
];

function listClassAssets(className) {
  const classDir = path.join(ASSET_BANK_DIR, className);
  if (!fs.existsSync(classDir)) return [];
  return fs.readdirSync(classDir)
    .filter((name) => /\.png$/i.test(name))
    .map((name) => path.join(classDir, name))
    .sort();
}

const assetPathPools = Object.fromEntries(CLASS_NAMES.map((className) => [className, listClassAssets(className)]));
for (const [className, paths] of Object.entries(assetPathPools)) {
  if (paths.length === 0) {
    throw new Error(`No scan assets found for ${className} under ${ASSET_BANK_DIR}`);
  }
}

const baseOccluders = [
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

function variantAssets(variant) {
  if (effectiveSceneMode === "qa3") return qa3Assets(variant);
  if (effectiveSceneMode === "fan") return fanAssets(variant);
  if (effectiveSceneMode === "clean") return cleanAssets(variant);
  if (variant === 0) return baseAssets;
  const rng = mulberry32(26053003 + variant * 101);
  const noteCount = 3 + variant % 4;
  const layerOrder = Array.from({ length: noteCount }, (_, index) => index).sort(() => rng() - 0.5);
  const anchors = [
    [-0.34, 0.02],
    [0.10, -0.04],
    [0.00, 0.24],
    [-0.22, -0.18],
    [0.28, 0.16],
    [-0.44, 0.18],
  ];
  return Array.from({ length: noteCount }, (_, index) => {
    const classIndex = classIndexFor(variant, index);
    const className = CLASS_NAMES[classIndex];
    const base = baseAssets[index % baseAssets.length];
    const anchor = anchors[index % anchors.length];
    const layer = layerOrder.indexOf(index);
    return {
      ...base,
      classIndex,
      className,
      idColor: INSTANCE_ID_COLORS[index],
      path: assetPathPools[className][(variant + index) % assetPathPools[className].length],
      physicalWidthMm: PHYSICAL_WIDTH_MM[className],
      position: [
        anchor[0] + randomBetween(rng, -0.10, 0.10),
        anchor[1] + randomBetween(rng, -0.10, 0.10),
        0.03 + layer * 0.045,
      ],
      rotation: [
        base.rotation[0] + randomBetween(rng, -0.05, 0.05),
        base.rotation[1] + randomBetween(rng, -0.05, 0.05),
        base.rotation[2] + randomBetween(rng, -0.34, 0.34),
      ],
      curl: 0.065 + randomBetween(rng, -0.020, 0.030),
      ripple: 0.0,
      roughness: randomBetween(rng, 0.70, 0.90),
      layer,
    };
  });
}

function variantOccluders(variant) {
  if (effectiveSceneMode === "qa3") return [];
  if (effectiveSceneMode === "clean") return [];
  if (effectiveSceneMode === "fan") return fanOccluders(variant);
  if (variant === 0) return baseOccluders;
  const rng = mulberry32(26054003 + variant * 131);
  return baseOccluders.map((occluder) => ({
    ...occluder,
    position: [
      occluder.position[0] + randomBetween(rng, -0.16, 0.16),
      occluder.position[1] + randomBetween(rng, -0.14, 0.14),
      occluder.position[2],
    ],
    rotation: [
      occluder.rotation[0] + randomBetween(rng, -0.08, 0.08),
      occluder.rotation[1] + randomBetween(rng, -0.08, 0.08),
      occluder.rotation[2] + randomBetween(rng, -0.38, 0.38),
    ],
  }));
}

function cleanAssets(variant) {
  const rng = mulberry32(26055003 + variant * 149);
  const noteCount = 1 + variant % 3;
  const anchors = [
    [-0.36, -0.16],
    [0.26, 0.02],
    [-0.02, 0.28],
  ];
  return Array.from({ length: noteCount }, (_, index) => {
    const classIndex = classIndexFor(variant, index);
    const className = CLASS_NAMES[classIndex];
    const base = baseAssets[index % baseAssets.length];
    const anchor = anchors[index];
    return {
      ...base,
      classIndex,
      className,
      idColor: INSTANCE_ID_COLORS[index],
      path: assetPathPools[className][(variant + index) % assetPathPools[className].length],
      physicalWidthMm: PHYSICAL_WIDTH_MM[className],
      position: [
        anchor[0] + randomBetween(rng, -0.04, 0.04),
        anchor[1] + randomBetween(rng, -0.04, 0.04),
        0.03 + index * 0.01,
      ],
      rotation: [
        randomBetween(rng, -0.045, 0.045),
        randomBetween(rng, -0.055, 0.055),
        randomBetween(rng, -0.20, 0.20),
      ],
      curl: 0.030 + randomBetween(rng, -0.010, 0.016),
      ripple: 0.0,
      roughness: randomBetween(rng, 0.72, 0.88),
      layer: index,
      clean: true,
    };
  });
}

function qa3Assets(variant) {
  const classes = ["KHR_2000", "KHR_500", "KHR_1000"];
  const placements = [
    {
      position: [-0.08, -0.26, 0.03],
      rotation: [0.02, -0.04, -0.74],
      curl: 0.045,
      layer: 0,
    },
    {
      position: [-0.10, 0.02, 0.08],
      rotation: [0.02, 0.02, 0.00],
      curl: 0.035,
      layer: 1,
    },
    {
      position: [0.02, 0.06, 0.13],
      rotation: [0.02, -0.05, 1.56],
      curl: 0.040,
      layer: 2,
    },
  ];
  return classes.map((className, index) => {
    const classIndex = CLASS_NAMES.indexOf(className);
    return {
      ...baseAssets[index % baseAssets.length],
      classIndex,
      className,
      idColor: INSTANCE_ID_COLORS[index],
      path: assetPathPools[className][variant % assetPathPools[className].length],
      physicalWidthMm: PHYSICAL_WIDTH_MM[className],
      roughness: 0.80,
      ripple: 0.0,
      ...placements[index],
    };
  });
}

function fanAssets(variant) {
  const rng = mulberry32(26056003 + variant * 173);
  const noteCount = 5 + variant % 3;
  const pivot = [
    -0.38 + randomBetween(rng, -0.05, 0.04),
    -0.22 + randomBetween(rng, -0.05, 0.05),
  ];
  const localPivot = [-0.48, -0.18];
  const spread = randomBetween(rng, 0.82, 1.22);
  const startAngle = randomBetween(rng, -0.18, 0.08);
  const layerOrder = Array.from({ length: noteCount }, (_, index) => index).sort(() => rng() - 0.5);
  return Array.from({ length: noteCount }, (_, index) => {
    const t = noteCount === 1 ? 0.5 : index / (noteCount - 1);
    const classIndex = classIndexFor(variant, index);
    const className = CLASS_NAMES[classIndex];
    const base = baseAssets[index % baseAssets.length];
    const layer = layerOrder.indexOf(index);
    const theta = startAngle + (t - 0.5) * spread + randomBetween(rng, -0.045, 0.045);
    const localPivotRotated = rotate2(localPivot, theta);
    const center = [
      pivot[0] - localPivotRotated[0] + randomBetween(rng, -0.012, 0.012),
      pivot[1] - localPivotRotated[1] + randomBetween(rng, -0.012, 0.012),
    ];
    return {
      ...base,
      classIndex,
      className,
      idColor: INSTANCE_ID_COLORS[index],
      path: assetPathPools[className][(variant + index) % assetPathPools[className].length],
      physicalWidthMm: PHYSICAL_WIDTH_MM[className],
      position: [
        center[0],
        center[1],
        0.03 + layer * 0.040,
      ],
      rotation: [
        randomBetween(rng, -0.12, 0.12),
        randomBetween(rng, -0.18, 0.18),
        theta,
      ],
      curl: 0.080 + randomBetween(rng, -0.025, 0.045),
      ripple: 0.0,
      roughness: randomBetween(rng, 0.70, 0.90),
      layer,
      fan: {
        pivot,
        localPivot,
        theta,
        spread,
      },
    };
  });
}

function fanOccluders(variant) {
  const rng = mulberry32(26057003 + variant * 181);
  const baseX = -0.38 + randomBetween(rng, -0.04, 0.04);
  const baseY = -0.24 + randomBetween(rng, -0.04, 0.04);
  return [
    {
      kind: "finger_capsule",
      layer: 20,
      color: 0xc58663,
      position: [baseX + 0.05, baseY + 0.10, 0.36],
      rotation: [0.16, -0.08, 1.18 + randomBetween(rng, -0.12, 0.12)],
      radius: 0.050,
      length: 0.46,
    },
    {
      kind: "finger_capsule",
      layer: 21,
      color: 0xb87958,
      position: [baseX + 0.15, baseY + 0.02, 0.37],
      rotation: [0.10, 0.16, 0.92 + randomBetween(rng, -0.12, 0.12)],
      radius: 0.046,
      length: 0.42,
    },
    {
      kind: "finger_capsule",
      layer: 22,
      color: 0xd09a77,
      position: [baseX - 0.02, baseY - 0.02, 0.38],
      rotation: [0.14, -0.12, 1.42 + randomBetween(rng, -0.10, 0.10)],
      radius: 0.042,
      length: 0.34,
    },
  ];
}

const assets = variantAssets(VARIANT);
const occluders = variantOccluders(VARIANT);
const backgroundFiles = listImageFiles(BACKGROUND_DIR);
const selectedBackgroundPath = backgroundFiles.length ? backgroundFiles[VARIANT % backgroundFiles.length] : null;
const config = sceneConfig(VARIANT, effectiveSceneMode, selectedBackgroundPath);

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
const sceneConfig = ${JSON.stringify(config)};
const WIDTH = ${JSON.stringify(WIDTH)};
const HEIGHT = ${JSON.stringify(HEIGHT)};
const VISUAL_SCALE = ${JSON.stringify(VISUAL_SCALE)};
const MIN_VISIBLE_PIXELS = ${JSON.stringify(MIN_VISIBLE_PIXELS)};
const scene = new THREE.Scene();
scene.background = new THREE.Color(sceneConfig.surface.scene);

const camera = new THREE.PerspectiveCamera(sceneConfig.camera.fov, WIDTH / HEIGHT, 0.01, 20);
camera.position.set(...sceneConfig.camera.position);
camera.lookAt(...sceneConfig.camera.lookAt);

const visualRenderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
visualRenderer.setPixelRatio(VISUAL_SCALE);
visualRenderer.setSize(WIDTH, HEIGHT);
visualRenderer.shadowMap.enabled = true;
visualRenderer.shadowMap.type = THREE.PCFShadowMap;
visualRenderer.domElement.style.position = "absolute";
visualRenderer.domElement.style.left = "0";
visualRenderer.domElement.style.top = "0";
document.body.appendChild(visualRenderer.domElement);

const idRenderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
idRenderer.setSize(WIDTH, HEIGHT);
idRenderer.shadowMap.enabled = false;

const grainCanvas = document.createElement("canvas");
grainCanvas.width = WIDTH;
grainCanvas.height = HEIGHT;
grainCanvas.style.position = "absolute";
grainCanvas.style.left = "0";
grainCanvas.style.top = "0";
grainCanvas.style.pointerEvents = "none";
grainCanvas.style.display = "none";
document.body.appendChild(grainCanvas);

function buildCameraOverlay() {
  const context = grainCanvas.getContext("2d", { willReadFrequently: true });
  const image = context.createImageData(grainCanvas.width, grainCanvas.height);
  const cx = grainCanvas.width / 2;
  const cy = grainCanvas.height / 2;
  const maxRadius = Math.sqrt(cx * cx + cy * cy);
  for (let y = 0; y < grainCanvas.height; y += 1) {
    for (let x = 0; x < grainCanvas.width; x += 1) {
      const i = (y * grainCanvas.width + x) * 4;
      const hash = Math.sin((x + 1) * 12.9898 + (y + 1) * 78.233) * 43758.5453;
      const grain = (hash - Math.floor(hash) - 0.5) * 38;
      const radius = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / maxRadius;
      const vignette = Math.max(0, radius - 0.48) * sceneConfig.postprocess.vignette;
      const value = Math.max(0, Math.min(255, 128 + grain - vignette));
      image.data[i] = value;
      image.data[i + 1] = value;
      image.data[i + 2] = value;
      image.data[i + 3] = sceneConfig.postprocess.grainAlpha;
    }
  }
  context.putImageData(image, 0, 0);
}

buildCameraOverlay();

const hemi = new THREE.HemisphereLight(0xf5f1e7, 0x6a5d50, sceneConfig.lighting.hemiIntensity);
scene.add(hemi);
const key = new THREE.DirectionalLight(sceneConfig.lighting.keyColor, sceneConfig.lighting.keyIntensity);
key.position.set(...sceneConfig.lighting.keyPosition);
key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
scene.add(key);

const loader = new THREE.TextureLoader();
const table = new THREE.Mesh(
  new THREE.PlaneGeometry(4.0, 3.0, 4, 4),
  new THREE.MeshStandardMaterial({ color: 0xffffff, map: makeTableTexture(), roughness: 0.88 })
);
table.receiveShadow = true;
table.position.z = -0.02;
scene.add(table);

const meshes = [];
const backingMeshes = [];
const occluderMeshes = [];

function makeTableTexture() {
  if (sceneConfig.surface.background) {
    const texture = loader.load(sceneConfig.surface.background.textureUrl);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(1.0, 1.0);
    texture.anisotropy = visualRenderer.capabilities.getMaxAnisotropy();
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.magFilter = THREE.LinearFilter;
    return texture;
  }

  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const context = canvas.getContext("2d");
  context.fillStyle = "#" + sceneConfig.surface.base.toString(16).padStart(6, "0");
  context.fillRect(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < canvas.height; y += 1) {
    const stripe = Math.sin(y * 0.09) * 9 + Math.sin(y * 0.021) * 18;
    const light = sceneConfig.surface.light;
    const lr = (light >> 16) & 255;
    const lg = (light >> 8) & 255;
    const lb = light & 255;
    context.fillStyle = "rgba(" + lr + "," + lg + "," + lb + ",0.035)";
    context.fillRect(0, y, canvas.width, Math.max(1, Math.abs(stripe) * 0.06));
  }
  for (let i = 0; i < 1800; i += 1) {
    const x = (Math.sin(i * 12.9898) * 43758.5453) % 1;
    const y = (Math.sin(i * 78.233) * 24634.6345) % 1;
    const px = Math.abs(x) * canvas.width;
    const py = Math.abs(y) * canvas.height;
    const alpha = 0.025 + (i % 7) * 0.003;
    const color = i % 2 === 0 ? sceneConfig.surface.light : sceneConfig.surface.dark;
    const r = (color >> 16) & 255;
    const g = (color >> 8) & 255;
    const b = color & 255;
    context.fillStyle = "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
    context.fillRect(px, py, 1, 1);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(...sceneConfig.surface.repeat);
  return texture;
}

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

function makeFingerGeometry(radius, length) {
  const geometry = new THREE.CapsuleGeometry(radius, length, 16, 48);
  geometry.computeVertexNormals();
  return geometry;
}

async function addNotes() {
  for (const asset of [...assets].sort((a, b) => a.layer - b.layer)) {
    const texture = await loader.loadAsync(asset.textureUrl);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = visualRenderer.capabilities.getMaxAnisotropy();
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.generateMipmaps = true;
    texture.needsUpdate = true;
    const aspect = texture.image.width / texture.image.height;
    const noteWidth = 1.28 * ((asset.physicalWidthMm ?? 156) / 156);
    const noteHeight = noteWidth / aspect;
    const ySegments = Math.max(48, Math.round(160 / aspect));
    const geometry = new THREE.PlaneGeometry(noteWidth, noteHeight, 160, ySegments);
    bendGeometry(geometry, asset.curl ?? 0.075, asset.ripple ?? 0.0);
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
    backing.receiveShadow = false;
    backing.userData = { material: backingMaterial };
    backingMeshes.push(backing);
    scene.add(backing);

    const material = new THREE.MeshStandardMaterial({
      map: texture,
      roughness: asset.roughness ?? 0.82,
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
    mesh.receiveShadow = false;
    mesh.userData = { material, idColor: asset.idColor, asset };
    meshes.push(mesh);
    scene.add(mesh);
  }
}

function addOccluders() {
  for (const occluder of occluders) {
    const geometry = makeFingerGeometry(occluder.radius, occluder.length);
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
        color: idColorFromBytes([r, g, b]),
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
    scene.background = new THREE.Color(sceneConfig.surface.scene);
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
  grainCanvas.style.display = mode === "id" ? "none" : "block";
  visualRenderer.domElement.style.filter = mode === "id" ? "none" : "contrast(" + sceneConfig.postprocess.contrast + ") saturate(" + sceneConfig.postprocess.saturation + ") brightness(" + sceneConfig.postprocess.brightness + ")";
  const activeRenderer = mode === "id" ? idRenderer : visualRenderer;
  activeRenderer.render(scene, camera);
};

function captureCanvasPixels(sourceRenderer = idRenderer) {
  const canvas = sourceRenderer.domElement;
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

function idColorFromBytes([r, g, b]) {
  const color = new THREE.Color();
  color.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
  return color;
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

  const colorToClass = new Map(assets.map((asset) => [asset.idColor.join(","), { classIndex: asset.classIndex, className: asset.className }]));
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
  })).filter((box) => box.pixels >= MIN_VISIBLE_PIXELS).sort((a, b) => a.classIndex - b.classIndex);
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
      color: idColorFromBytes([r, g, b]),
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide
    });
    idRenderer.render(scene, camera);
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
window.captureIdPng = () => idRenderer.domElement.toDataURL("image/png");
window.__cashsnapReady = true;
</script>
</body>
</html>`;
}

function writeDataUrlPng(dataUrl, outPath) {
  const prefix = "data:image/png;base64,";
  if (!dataUrl.startsWith(prefix)) {
    throw new Error("Expected PNG data URL");
  }
  fs.writeFileSync(outPath, Buffer.from(dataUrl.slice(prefix.length), "base64"));
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
    page.setDefaultTimeout(180000);
    page.setDefaultNavigationTimeout(180000);
    await page.setViewport({ width: WIDTH, height: HEIGHT, deviceScaleFactor: 1 });
    const htmlPath = path.join(OUT_DIR, "smoke.html");
    fs.writeFileSync(htmlPath, html(textureAssets));
    await page.goto(pathToFileURL(htmlPath).href, { waitUntil: "domcontentloaded", timeout: 180000 });
    await page.waitForFunction("window.__cashsnapReady === true");
    await page.evaluate(() => window.renderPass("visual"));
    await page.screenshot({ path: path.join(OUT_DIR, "visual.png") });
    const idPngDataUrl = await page.evaluate(() => {
      window.renderPass("id");
      return window.captureIdPng();
    });
    writeDataUrlPng(idPngDataUrl, path.join(OUT_DIR, "id.png"));
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
        variant: VARIANT,
        sceneMode: effectiveSceneMode,
        width: WIDTH,
        height: HEIGHT,
        visualScale: VISUAL_SCALE,
        minVisiblePixels: MIN_VISIBLE_PIXELS,
        sceneConfig: config,
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
