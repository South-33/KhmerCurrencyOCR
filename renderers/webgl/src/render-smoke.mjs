import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import puppeteer from "puppeteer-core";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..", "..", "..");
const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const THREE_MODULE = pathToFileURL(path.join(ROOT, "renderers", "webgl", "node_modules", "three", "build", "three.module.js")).href;
const HDR_LOADER_MODULE = pathToFileURL(path.join(ROOT, "renderers", "webgl", "node_modules", "three", "examples", "jsm", "loaders", "HDRLoader.js")).href;

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
const ENVIRONMENT_DIR = argValue("--environment-dir", "");
const ASSET_SIDE_POLICY = argValue("--asset-side-policy", "any");
const CAMERA_PROFILE = argValue("--camera-profile", "generic_phone_jitter");
const CLASS_SEQUENCE_RAW = argValue("--class-sequence", "");
const NOTE_CONDITION_POLICY = argValue("--note-condition-policy", "mixed");
const BROWSER_EXECUTABLE = argValue("--browser-executable", process.env.CASHSNAP_WEBGL_BROWSER || EDGE);
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

function parseClassSequence(value) {
  const classes = value.split(/[,;\s]+/).map((item) => item.trim()).filter(Boolean);
  const unknown = classes.filter((className) => !CLASS_NAMES.includes(className));
  if (unknown.length > 0) {
    throw new Error(`--class-sequence contains unknown class(es): ${unknown.join(", ")}`);
  }
  return classes;
}

const CLASS_SEQUENCE = parseClassSequence(CLASS_SEQUENCE_RAW);

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

if (!["auto", "clean", "clean_single", "negative", "stack", "fan", "thin_edge", "hand_occlusion", "qa3"].includes(SCENE_MODE)) {
  throw new Error("--scene-mode must be one of: auto, clean, clean_single, negative, stack, fan, thin_edge, hand_occlusion, qa3");
}

if (!["any", "front_only", "back_only", "front_back_mix"].includes(ASSET_SIDE_POLICY)) {
  throw new Error("--asset-side-policy must be one of: any, front_only, back_only, front_back_mix");
}

if (![
  "generic_phone_jitter",
  "phone_auto",
  "iphone_8_like",
  "iphone_12_wide_like",
  "budget_android_wide_like",
  "browser_upload_resized",
  "phone_closeup_clean_like",
  "phone_top_down_like",
  "phone_oblique_30_like",
  "phone_oblique_45_like",
  "phone_low_front_like",
].includes(CAMERA_PROFILE)) {
  throw new Error("--camera-profile must be one of: generic_phone_jitter, phone_auto, iphone_8_like, iphone_12_wide_like, budget_android_wide_like, browser_upload_resized, phone_closeup_clean_like, phone_top_down_like, phone_oblique_30_like, phone_oblique_45_like, phone_low_front_like");
}

if (!["mixed", "pristine_only", "heavy_wear", "wet_stress"].includes(NOTE_CONDITION_POLICY)) {
  throw new Error("--note-condition-policy must be one of: mixed, pristine_only, heavy_wear, wet_stress");
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

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function round3(value) {
  return Math.round(value * 1000) / 1000;
}

function rotate2([x, y], angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [x * c - y * s, x * s + y * c];
}

function noteCountForSequenceMode(mode, variant) {
  if (mode === "negative") return 0;
  if (mode === "clean_single") return 1;
  if (mode === "clean") return 1 + variant % 3;
  if (mode === "qa3") return 3;
  if (mode === "thin_edge") return 4;
  if (mode === "hand_occlusion") return 4 + variant % 2;
  if (mode === "fan") return 5 + variant % 3;
  return variant === 0 ? 3 : 3 + variant % 4;
}

function classSequenceOffset(variant) {
  let offset = 0;
  for (let previous = 0; previous < variant; previous += 1) {
    offset += noteCountForSequenceMode(effectiveSceneMode, previous);
  }
  return offset;
}

function classSequenceIndex(variant, index) {
  return (classSequenceOffset(variant) + index) % CLASS_SEQUENCE.length;
}

function classIndexFor(variant, index) {
  if (CLASS_SEQUENCE.length > 0) {
    const className = CLASS_SEQUENCE[classSequenceIndex(variant, index)];
    return CLASS_NAMES.indexOf(className);
  }
  return (variant * 7 + index * 5) % CLASS_NAMES.length;
}

function classNameForSequenceOrFallback(variant, index, fallbackClasses) {
  if (CLASS_SEQUENCE.length > 0) {
    return CLASS_SEQUENCE[classSequenceIndex(variant, index)];
  }
  return fallbackClasses[(variant + index) % fallbackClasses.length];
}

function noteConditionFor(asset, variant, index) {
  const seed = 26056503 + variant * 1009 + index * 9173 + (asset.classIndex ?? 0) * 137;
  const rng = mulberry32(seed);
  const cleanScene = Boolean(asset.clean || asset.cleanSingle || effectiveSceneMode === "clean" || effectiveSceneMode === "clean_single");
  let dirtiness = 0;
  let crinkle = 0;
  let wetness = 0;
  if (NOTE_CONDITION_POLICY === "pristine_only") {
    dirtiness = randomBetween(rng, 0.00, 0.08);
    crinkle = randomBetween(rng, 0.00, 0.14);
  } else if (NOTE_CONDITION_POLICY === "heavy_wear") {
    dirtiness = randomBetween(rng, 0.55, 1.00);
    crinkle = randomBetween(rng, 0.34, 1.00);
    wetness = rng() < 0.22 ? randomBetween(rng, 0.10, 0.52) : 0.0;
  } else if (NOTE_CONDITION_POLICY === "wet_stress") {
    dirtiness = randomBetween(rng, 0.18, 0.82);
    crinkle = randomBetween(rng, 0.10, 0.72);
    wetness = randomBetween(rng, 0.32, 0.90);
  } else {
    const bucket = rng();
    dirtiness = bucket < 0.18
      ? randomBetween(rng, 0.00, 0.12)
      : bucket < 0.74
        ? randomBetween(rng, 0.12, 0.58)
        : randomBetween(rng, 0.58, 1.00);
    crinkle = Math.pow(rng(), 1.25);
    if (rng() < 0.18) crinkle = randomBetween(rng, 0.0, 0.16);
    if (cleanScene) {
      dirtiness *= 0.45;
      crinkle *= 0.42;
    }
    const wetChance = cleanScene ? 0.06 : 0.18;
    wetness = rng() < wetChance ? randomBetween(rng, cleanScene ? 0.06 : 0.12, cleanScene ? 0.28 : 0.78) : 0.0;
  }
  const edgeWear = clamp(dirtiness * randomBetween(rng, 0.40, 1.10) + crinkle * 0.20, 0, 1);
  const creaseCount = Math.max(0, Math.round(crinkle * randomBetween(rng, 1.0, 5.0)));
  const stainCount = Math.max(0, Math.round(dirtiness * randomBetween(rng, 1.0, 4.5) + wetness * 3.0));
  const speckleCount = Math.max(0, Math.round(12 + dirtiness * randomBetween(rng, 80, 210)));
  const profile = wetness > 0.42
    ? "damp"
    : dirtiness < 0.12 && crinkle < 0.14
      ? "pristine"
      : dirtiness > 0.70 || crinkle > 0.72
        ? "heavily_circulated"
        : "circulated";
  return {
    profile,
    policy: NOTE_CONDITION_POLICY,
    seed,
    dirtiness: round3(dirtiness),
    crinkle: round3(crinkle),
    wetness: round3(wetness),
    edgeWear: round3(edgeWear),
    creaseCount,
    stainCount,
    speckleCount,
    creaseAngle: round3(randomBetween(rng, -1.4, 1.4)),
    creaseOffset: round3(randomBetween(rng, -0.28, 0.28)),
    wavePhase: round3(randomBetween(rng, 0, Math.PI * 2)),
  };
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

function listEnvironmentFiles(directory) {
  if (!directory) return [];
  const resolved = path.resolve(ROOT, directory);
  if (!fs.existsSync(resolved)) return [];
  return fs.readdirSync(resolved)
    .filter((name) => /\.(hdr|png|jpe?g|webp)$/i.test(name))
    .map((name) => path.join(resolved, name))
    .sort();
}

const CAMERA_PROFILES = {
  generic_phone_jitter: {
    source: "legacy_webgl_default",
    weight: 1.0,
    targetResolution: [1440, 1080],
    fov: [42, 56],
    positionX: [-0.10, 0.10],
    positionY: [-0.48, -0.24],
    positionZ: [2.05, 2.42],
    lookAtX: [-0.05, 0.05],
    lookAtY: [-0.03, 0.05],
  },
  iphone_8_like: {
    source: "configs/3d_pipeline/proof_p1_transfer.json",
    weight: 0.25,
    targetResolution: [960, 720],
    fov: [48, 64],
    positionX: [-0.12, 0.12],
    positionY: [-0.58, -0.28],
    positionZ: [2.10, 2.55],
    lookAtX: [-0.06, 0.06],
    lookAtY: [-0.04, 0.06],
  },
  iphone_12_wide_like: {
    source: "configs/3d_pipeline/proof_p1_transfer.json",
    weight: 0.35,
    targetResolution: [960, 720],
    fov: [58, 74],
    positionX: [-0.14, 0.14],
    positionY: [-0.55, -0.22],
    positionZ: [1.85, 2.32],
    lookAtX: [-0.07, 0.07],
    lookAtY: [-0.05, 0.07],
  },
  budget_android_wide_like: {
    source: "configs/3d_pipeline/proof_p1_transfer.json",
    weight: 0.25,
    targetResolution: [960, 720],
    fov: [62, 80],
    positionX: [-0.15, 0.15],
    positionY: [-0.58, -0.20],
    positionZ: [1.75, 2.25],
    lookAtX: [-0.08, 0.08],
    lookAtY: [-0.05, 0.08],
  },
  browser_upload_resized: {
    source: "configs/3d_pipeline/proof_p1_transfer.json",
    weight: 0.15,
    targetResolution: [640, 640],
    fov: [50, 70],
    positionX: [-0.10, 0.10],
    positionY: [-0.44, -0.18],
    positionZ: [1.95, 2.35],
    lookAtX: [-0.05, 0.05],
    lookAtY: [-0.03, 0.05],
  },
  phone_closeup_clean_like: {
    source: "cashsnap_domain_gap_audit_v1",
    weight: 0.0,
    targetResolution: [1440, 1080],
    fov: [40, 54],
    positionX: [-0.10, 0.10],
    positionY: [-0.30, -0.06],
    positionZ: [1.45, 1.70],
    lookAtX: [-0.05, 0.05],
    lookAtY: [-0.03, 0.05],
  },
  phone_top_down_like: {
    source: "cashsnap_viewpoint_v1",
    weight: 0.20,
    targetResolution: [1440, 1080],
    fov: [48, 64],
    positionX: [-0.12, 0.12],
    positionY: [-0.22, 0.10],
    positionZ: [2.10, 2.65],
    lookAtX: [-0.05, 0.05],
    lookAtY: [-0.04, 0.06],
  },
  phone_oblique_30_like: {
    source: "cashsnap_viewpoint_v1",
    weight: 0.30,
    targetResolution: [1440, 1080],
    fov: [52, 68],
    positionX: [-0.18, 0.18],
    positionY: [-0.90, -0.55],
    positionZ: [1.55, 2.05],
    lookAtX: [-0.06, 0.06],
    lookAtY: [-0.02, 0.12],
  },
  phone_oblique_45_like: {
    source: "cashsnap_viewpoint_v1",
    weight: 0.25,
    targetResolution: [1440, 1080],
    fov: [58, 76],
    positionX: [-0.22, 0.22],
    positionY: [-1.45, -0.95],
    positionZ: [1.25, 1.75],
    lookAtX: [-0.08, 0.08],
    lookAtY: [0.00, 0.18],
  },
  phone_low_front_like: {
    source: "cashsnap_viewpoint_v1",
    weight: 0.12,
    targetResolution: [1440, 1080],
    fov: [64, 82],
    positionX: [-0.25, 0.25],
    positionY: [-1.95, -1.35],
    positionZ: [0.95, 1.35],
    lookAtX: [-0.10, 0.10],
    lookAtY: [0.08, 0.28],
  },
};

function chooseCameraProfile(rng, requestedProfile) {
  if (requestedProfile !== "phone_auto") {
    return { name: requestedProfile, ...CAMERA_PROFILES[requestedProfile] };
  }
  const candidates = [
    "phone_top_down_like",
    "phone_oblique_30_like",
    "phone_oblique_45_like",
    "phone_low_front_like",
    "iphone_12_wide_like",
    "budget_android_wide_like",
    "browser_upload_resized",
  ];
  const totalWeight = candidates.reduce((total, name) => total + CAMERA_PROFILES[name].weight, 0);
  let threshold = rng() * totalWeight;
  for (const name of candidates) {
    threshold -= CAMERA_PROFILES[name].weight;
    if (threshold <= 0) return { name, ...CAMERA_PROFILES[name] };
  }
  const fallback = candidates[candidates.length - 1];
  return { name: fallback, ...CAMERA_PROFILES[fallback] };
}

function cameraViewAngles(position, lookAt) {
  const dx = position[0] - lookAt[0];
  const dy = position[1] - lookAt[1];
  const dz = position[2] - lookAt[2];
  const horizontalDistance = Math.hypot(dx, dy);
  const verticalDistance = Math.abs(dz);
  const fromVertical = Math.atan2(horizontalDistance, verticalDistance) * 180 / Math.PI;
  const aboveTable = Math.atan2(verticalDistance, horizontalDistance || 1e-6) * 180 / Math.PI;
  return {
    fromVerticalDeg: Number(fromVertical.toFixed(2)),
    aboveTableDeg: Number(aboveTable.toFixed(2)),
  };
}

function sceneConfig(variant, mode, backgroundPath, environmentPath) {
  const modeOffset = mode === "fan" ? 1009 : mode === "clean" ? 2003 : mode === "clean_single" ? 2309 : mode === "qa3" ? 3001 : mode === "negative" ? 4001 : mode === "thin_edge" ? 5003 : mode === "hand_occlusion" ? 6007 : 0;
  const rng = mulberry32(26058003 + variant * 191 + modeOffset);
  const cameraProfile = chooseCameraProfile(rng, CAMERA_PROFILE);
  const cameraPosition = [
    randomBetween(rng, ...cameraProfile.positionX),
    randomBetween(rng, ...cameraProfile.positionY),
    randomBetween(rng, ...cameraProfile.positionZ),
  ];
  const cameraLookAt = [
    randomBetween(rng, ...cameraProfile.lookAtX),
    randomBetween(rng, ...cameraProfile.lookAtY),
    0,
  ];
  const cameraAngles = cameraViewAngles(cameraPosition, cameraLookAt);
  const defaultSurfaces = [
    { name: "warm_wood", base: "#9b784a", light: "#fff2d6", dark: "#231810", scene: "#9b927d", repeat: [2.5, 2.0] },
    { name: "gray_counter", base: "#827f77", light: "#dedbd2", dark: "#3a3834", scene: "#86837b", repeat: [1.8, 1.8] },
    { name: "green_plastic_mat", base: "#456f5b", light: "#b7d8c5", dark: "#1c3229", scene: "#506f61", repeat: [2.0, 1.7] },
    { name: "blue_shop_table", base: "#566f83", light: "#d6e4ef", dark: "#25323b", scene: "#627487", repeat: [2.1, 1.9] },
    { name: "dark_laminate", base: "#51473c", light: "#b5a28a", dark: "#1f1a15", scene: "#5a5147", repeat: [2.8, 2.2] },
  ];
  const cleanSingleSurfaces = [
    { name: "off_white_counter", base: "#c8c5ba", light: "#fffdf2", dark: "#77736a", scene: "#cfcbc0", repeat: [1.9, 1.7] },
    { name: "pale_shop_tile", base: "#bbb8ad", light: "#f7f5ea", dark: "#6f6b62", scene: "#c5c1b6", repeat: [2.2, 2.2] },
    { name: "light_gray_table", base: "#aaa9a2", light: "#efeee7", dark: "#62615b", scene: "#b6b5ae", repeat: [2.0, 1.8] },
    { name: "cool_countertop", base: "#aeb7b5", light: "#edf3ef", dark: "#68716f", scene: "#bbc4c2", repeat: [1.8, 1.8] },
  ];
  const surfaces = mode === "clean_single" ? cleanSingleSurfaces : defaultSurfaces;
  const surface = surfaces[randomInt(rng, surfaces.length)];
  const cleanSingle = mode === "clean_single";
  const environmentExtension = environmentPath ? path.extname(environmentPath).toLowerCase() : "";
  return {
    noteConditionPolicy: NOTE_CONDITION_POLICY,
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
    environment: environmentPath ? {
      path: environmentPath,
      textureUrl: pathToFileURL(environmentPath).href,
      format: environmentExtension === ".hdr" ? "hdr_equirectangular" : "ldr_equirectangular",
      mapping: "EquirectangularReflectionMapping",
      role: "visual_lighting_reflection_only_id_pass_uses_black_background",
    } : null,
    camera: {
      profileRequested: CAMERA_PROFILE,
      profile: cameraProfile.name,
      profileSource: cameraProfile.source,
      targetResolution: cameraProfile.targetResolution,
      fov: randomBetween(rng, ...cameraProfile.fov),
      position: cameraPosition,
      lookAt: cameraLookAt,
      viewAngleFromVerticalDeg: cameraAngles.fromVerticalDeg,
      viewAngleAboveTableDeg: cameraAngles.aboveTableDeg,
      lensDistortion: "not_applied_until_rgb_id_and_labels_share_the_same_exact_transform",
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
      contrast: cleanSingle ? randomBetween(rng, 1.06, 1.14) : randomBetween(rng, 1.00, 1.07),
      saturation: cleanSingle ? randomBetween(rng, 1.25, 1.45) : randomBetween(rng, 0.90, 1.07),
      brightness: cleanSingle ? randomBetween(rng, 0.92, 1.00) : randomBetween(rng, 0.96, 1.05),
      focusBlurPx: rng() < (cleanSingle ? 0.15 : 0.68) ? randomBetween(rng, cleanSingle ? 0.01 : 0.05, cleanSingle ? 0.06 : 0.32) : 0,
      grainStrength: randomBetween(rng, cleanSingle ? 42 : 24, cleanSingle ? 70 : 46),
      grainAlpha: randomBetween(rng, cleanSingle ? 20 : 14, cleanSingle ? 36 : 30),
      vignette: randomBetween(rng, cleanSingle ? 18 : 42, cleanSingle ? 44 : 78),
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

function sideForAssetPath(assetPath) {
  const name = path.basename(assetPath).toLowerCase();
  if (name.includes("_front")) return "front";
  if (name.includes("_back")) return "back";
  return "unknown";
}

const assetPathPoolsBySide = Object.fromEntries(
  CLASS_NAMES.map((className) => {
    const paths = assetPathPools[className];
    return [
      className,
      {
        front: paths.filter((assetPath) => sideForAssetPath(assetPath) === "front"),
        back: paths.filter((assetPath) => sideForAssetPath(assetPath) === "back"),
        unknown: paths.filter((assetPath) => sideForAssetPath(assetPath) === "unknown"),
      },
    ];
  }),
);

function targetSideForAsset(index) {
  if (ASSET_SIDE_POLICY === "front_only") return "front";
  if (ASSET_SIDE_POLICY === "back_only") return "back";
  if (ASSET_SIDE_POLICY === "front_back_mix") return index % 2 === 0 ? "front" : "back";
  return "any";
}

function selectAssetPath(className, variant, index) {
  const targetSide = targetSideForAsset(index);
  const pool = targetSide === "any" ? assetPathPools[className] : assetPathPoolsBySide[className][targetSide];
  if (!pool || pool.length === 0) {
    throw new Error(`No ${targetSide} scan assets for ${className}; cannot satisfy asset side policy ${ASSET_SIDE_POLICY}`);
  }
  return pool[(variant + index) % pool.length];
}

function enrichAsset(asset, variant, index) {
  const assetPath = selectAssetPath(asset.className, variant, index);
  return {
    ...asset,
    path: assetPath,
    side: sideForAssetPath(assetPath),
    assetSidePolicy: ASSET_SIDE_POLICY,
    condition: noteConditionFor(asset, variant, index),
  };
}

function annotateAsset(asset, index) {
  return {
    ...asset,
    side: sideForAssetPath(asset.path),
    assetSidePolicy: ASSET_SIDE_POLICY,
    condition: noteConditionFor(asset, 0, index),
  };
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
  if (effectiveSceneMode === "negative") return [];
  if (effectiveSceneMode === "qa3") return qa3Assets(variant);
  if (effectiveSceneMode === "fan") return fanAssets(variant);
  if (effectiveSceneMode === "thin_edge") return thinEdgeAssets(variant);
  if (effectiveSceneMode === "hand_occlusion") return handOcclusionAssets(variant);
  if (effectiveSceneMode === "clean_single") return cleanSingleAssets(variant);
  if (effectiveSceneMode === "clean") return cleanAssets(variant);
  if (variant === 0 && ASSET_SIDE_POLICY === "any") return baseAssets.map((asset, index) => annotateAsset(asset, index));
  if (variant === 0) return baseAssets.map((asset, index) => enrichAsset(asset, variant, index));
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
    return enrichAsset({
      ...base,
      classIndex,
      className,
      idColor: INSTANCE_ID_COLORS[index],
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
    }, variant, index);
  });
}

function variantOccluders(variant) {
  if (effectiveSceneMode === "negative") return negativeOccluders(variant);
  if (effectiveSceneMode === "qa3") return [];
  if (effectiveSceneMode === "clean" || effectiveSceneMode === "clean_single") return [];
  if (effectiveSceneMode === "fan") return fanOccluders(variant);
  if (effectiveSceneMode === "thin_edge") return thinEdgeOccluders(variant);
  if (effectiveSceneMode === "hand_occlusion") return handOcclusionOccluders(variant);
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
    return enrichAsset({
      ...base,
      classIndex,
      className,
      idColor: INSTANCE_ID_COLORS[index],
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
    }, variant, index);
  });
}

function cleanSingleAssets(variant) {
  const rng = mulberry32(26055291 + variant * 157);
  const classIndex = classIndexFor(variant, 0);
  const className = CLASS_NAMES[classIndex];
  const zRange = cleanSingleZRangeForClass(className);
  return [
    enrichAsset({
      ...baseAssets[0],
      classIndex,
      className,
      idColor: INSTANCE_ID_COLORS[0],
      physicalWidthMm: PHYSICAL_WIDTH_MM[className],
      position: [
        randomBetween(rng, -0.035, 0.035),
        randomBetween(rng, -0.030, 0.030),
        0.045,
      ],
      rotation: [
        randomBetween(rng, -0.035, 0.055),
        randomBetween(rng, -0.055, 0.055),
        randomBetween(rng, zRange[0], zRange[1]),
      ],
      curl: 0.026 + randomBetween(rng, -0.006, 0.010),
      ripple: 0.0,
      roughness: randomBetween(rng, 0.76, 0.90),
      layer: 0,
      cleanSingle: true,
    }, variant, 0),
  ];
}

function cleanSingleZRangeForClass(className) {
  const khrRanges = {
    KHR_500: [-0.12, 0.12],
    KHR_2000: [-0.34, 0.34],
    KHR_5000: [-0.22, 0.22],
    KHR_1000: [-0.42, 0.42],
    KHR_10000: [-0.12, 0.12],
    KHR_20000: [-0.24, 0.24],
    KHR_50000: [-0.20, 0.20],
  };
  return khrRanges[className] || [-0.62, 0.62];
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
    return enrichAsset({
      ...baseAssets[index % baseAssets.length],
      classIndex,
      className,
      idColor: INSTANCE_ID_COLORS[index],
      physicalWidthMm: PHYSICAL_WIDTH_MM[className],
      roughness: 0.80,
      ripple: 0.0,
      ...placements[index],
    }, variant, index);
  });
}

function thinEdgeLayout(variant) {
  const rng = mulberry32(26058003 + variant * 197);
  const classes = ["KHR_5000", "KHR_20000", "KHR_10000", "KHR_500"];
  const placements = [
    {
      position: [-0.32, -0.22, 0.03],
      rotation: [0.04, -0.04, -0.18],
      coverOffset: [0.14, 0.00],
      coverSize: [1.18, 0.62],
    },
    {
      position: [0.30, -0.13, 0.06],
      rotation: [0.02, 0.05, 0.30],
      coverOffset: [-0.15, 0.00],
      coverSize: [1.18, 0.62],
    },
    {
      position: [-0.08, 0.24, 0.09],
      rotation: [0.02, -0.03, 1.42],
      coverOffset: [0.00, -0.12],
      coverSize: [1.18, 0.58],
    },
    {
      position: [0.42, 0.23, 0.12],
      rotation: [0.03, 0.02, -0.66],
      coverOffset: [0.08, 0.10],
      coverSize: [1.04, 0.54],
    },
  ];
  return placements.map((placement, index) => {
    const className = classNameForSequenceOrFallback(variant, index, classes);
    const classIndex = CLASS_NAMES.indexOf(className);
    const jitter = rotate2(
      [randomBetween(rng, -0.025, 0.025), randomBetween(rng, -0.025, 0.025)],
      placement.rotation[2],
    );
    return {
      classIndex,
      className,
      position: [
        placement.position[0] + jitter[0],
        placement.position[1] + jitter[1],
        placement.position[2],
      ],
      rotation: [
        placement.rotation[0] + randomBetween(rng, -0.025, 0.025),
        placement.rotation[1] + randomBetween(rng, -0.025, 0.025),
        placement.rotation[2] + randomBetween(rng, -0.07, 0.07),
      ],
      coverOffset: [
        placement.coverOffset[0] + randomBetween(rng, -0.020, 0.020),
        placement.coverOffset[1] + randomBetween(rng, -0.020, 0.020),
      ],
      coverSize: placement.coverSize,
      layer: index,
    };
  });
}

function thinEdgeAssets(variant) {
  return thinEdgeLayout(variant).map((item, index) => {
    const base = baseAssets[index % baseAssets.length];
    return enrichAsset({
      ...base,
      classIndex: item.classIndex,
      className: item.className,
      idColor: INSTANCE_ID_COLORS[index],
      physicalWidthMm: PHYSICAL_WIDTH_MM[item.className],
      position: item.position,
      rotation: item.rotation,
      curl: 0.040,
      ripple: 0.0,
      roughness: 0.82,
      layer: item.layer,
      thinEdge: true,
    }, variant, index);
  });
}

function thinEdgeOccluders(variant) {
  const colors = [0xe4dfd3, 0xcfc7b8, 0xf2eee4, 0xd8d0c2];
  return thinEdgeLayout(variant).map((item, index) => {
    const [dx, dy] = rotate2(item.coverOffset, item.rotation[2]);
    return {
      kind: "cover_card",
      layer: 30 + index,
      color: colors[index % colors.length],
      position: [item.position[0] + dx, item.position[1] + dy, item.position[2] + 0.055],
      rotation: [item.rotation[0], item.rotation[1], item.rotation[2]],
      width: item.coverSize[0],
      height: item.coverSize[1],
    };
  });
}

function handOcclusionAssets(variant) {
  const rng = mulberry32(26059003 + variant * 211);
  const classes = ["KHR_500", "KHR_1000", "KHR_5000", "KHR_20000", "USD_1"];
  const placements = [
    [-0.34, -0.18, -0.42],
    [0.12, -0.08, 0.10],
    [-0.06, 0.20, 0.62],
    [0.36, 0.18, -0.20],
    [-0.44, 0.26, 0.94],
  ];
  const noteCount = 4 + (variant % 2);
  return Array.from({ length: noteCount }, (_, index) => {
    const className = classNameForSequenceOrFallback(variant, index, classes);
    const classIndex = CLASS_NAMES.indexOf(className);
    const base = baseAssets[index % baseAssets.length];
    const placement = placements[index];
    return enrichAsset({
      ...base,
      classIndex,
      className,
      idColor: INSTANCE_ID_COLORS[index],
      physicalWidthMm: PHYSICAL_WIDTH_MM[className],
      position: [
        placement[0] + randomBetween(rng, -0.055, 0.055),
        placement[1] + randomBetween(rng, -0.045, 0.045),
        0.03 + index * 0.035,
      ],
      rotation: [
        randomBetween(rng, -0.06, 0.08),
        randomBetween(rng, -0.08, 0.08),
        placement[2] + randomBetween(rng, -0.16, 0.16),
      ],
      curl: 0.055 + randomBetween(rng, -0.012, 0.018),
      ripple: 0.0,
      roughness: randomBetween(rng, 0.74, 0.90),
      layer: index,
      handOcclusion: true,
    }, variant, index);
  });
}

function handOcclusionOccluders(variant) {
  const rng = mulberry32(26060003 + variant * 223);
  const baseX = -0.08 + randomBetween(rng, -0.05, 0.05);
  const baseY = -0.02 + randomBetween(rng, -0.04, 0.04);
  const skinColors = [0xc58663, 0xb87958, 0xd19a73, 0x9f6649, 0xe0ae87];
  const fingers = [
    { offset: [-0.28, 0.05], angle: 1.34, radius: 0.045, length: 0.62 },
    { offset: [-0.08, 0.06], angle: 1.12, radius: 0.050, length: 0.66 },
    { offset: [0.12, 0.03], angle: 0.92, radius: 0.047, length: 0.60 },
    { offset: [0.28, -0.02], angle: 0.74, radius: 0.041, length: 0.52 },
    { offset: [-0.18, -0.16], angle: 1.70, radius: 0.043, length: 0.48 },
  ];
  return fingers.map((finger, index) => ({
    kind: "finger_capsule",
    layer: 35 + index,
    color: skinColors[(variant + index) % skinColors.length],
    position: [
      baseX + finger.offset[0] + randomBetween(rng, -0.035, 0.035),
      baseY + finger.offset[1] + randomBetween(rng, -0.030, 0.030),
      0.32 + index * 0.012,
    ],
    rotation: [
      randomBetween(rng, 0.06, 0.20),
      randomBetween(rng, -0.16, 0.16),
      finger.angle + randomBetween(rng, -0.18, 0.18),
    ],
    radius: finger.radius + randomBetween(rng, -0.006, 0.006),
    length: finger.length + randomBetween(rng, -0.040, 0.055),
  }));
}

function negativeOccluders(variant) {
  const rng = mulberry32(26057511 + variant * 149);
  const propCount = 2 + (variant % 3);
  const colors = [0xf2ead8, 0xd7e3ec, 0x24384d, 0xc8bca9, 0xe6d1a3, 0x5f6f61];
  const props = [];
  for (let index = 0; index < propCount; index += 1) {
    const useFinger = index === propCount - 1 && variant % 4 === 0;
    if (useFinger) {
      props.push({
        kind: "finger_capsule",
        layer: 30 + index,
        color: [0xc58663, 0xb87958, 0x9f6b4e][randomInt(rng, 3)],
        position: [
          randomBetween(rng, -0.55, 0.55),
          randomBetween(rng, -0.36, 0.36),
          0.08 + index * 0.012,
        ],
        rotation: [
          randomBetween(rng, -0.05, 0.20),
          randomBetween(rng, -0.12, 0.12),
          randomBetween(rng, -1.45, 1.45),
        ],
        radius: randomBetween(rng, 0.035, 0.055),
        length: randomBetween(rng, 0.35, 0.72),
      });
      continue;
    }
    props.push({
      kind: "cover_card",
      layer: 20 + index,
      color: colors[randomInt(rng, colors.length)],
      position: [
        randomBetween(rng, -0.62, 0.62),
        randomBetween(rng, -0.42, 0.42),
        0.05 + index * 0.01,
      ],
      rotation: [
        randomBetween(rng, -0.10, 0.10),
        randomBetween(rng, -0.10, 0.10),
        randomBetween(rng, -1.55, 1.55),
      ],
      width: randomBetween(rng, 0.28, 0.92),
      height: randomBetween(rng, 0.08, 0.32),
    });
  }
  return props;
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
    return enrichAsset({
      ...base,
      classIndex,
      className,
      idColor: INSTANCE_ID_COLORS[index],
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
    }, variant, index);
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
const environmentFiles = listEnvironmentFiles(ENVIRONMENT_DIR);
const selectedEnvironmentPath = environmentFiles.length ? environmentFiles[VARIANT % environmentFiles.length] : null;
const config = sceneConfig(VARIANT, effectiveSceneMode, selectedBackgroundPath, selectedEnvironmentPath);
const assetSideCounts = assets.reduce((counts, asset) => {
  const side = asset.side ?? "unknown";
  counts[side] = (counts[side] ?? 0) + 1;
  return counts;
}, {});
const frontBackMixSatisfied = assetSideCounts.front > 0 && assetSideCounts.back > 0;

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
import { HDRLoader } from "${HDR_LOADER_MODULE}";

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
      const grain = (hash - Math.floor(hash) - 0.5) * sceneConfig.postprocess.grainStrength;
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
async function loadEnvironmentTexture() {
  if (!sceneConfig.environment) return null;
  const texture = sceneConfig.environment.format === "hdr_equirectangular"
    ? await new HDRLoader().loadAsync(sceneConfig.environment.textureUrl)
    : await loader.loadAsync(sceneConfig.environment.textureUrl);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  if (sceneConfig.environment.format === "ldr_equirectangular") {
    texture.colorSpace = THREE.SRGBColorSpace;
  }
  return texture;
}

const environmentTexture = await loadEnvironmentTexture();
if (environmentTexture) {
  scene.environment = environmentTexture;
}

const table = new THREE.Mesh(
  new THREE.PlaneGeometry(30.0, 20.0, 8, 8),
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

function bendGeometry(geometry, curl, ripple, condition = {}) {
  const pos = geometry.attributes.position;
  const crinkle = condition.crinkle ?? 0.0;
  const wetness = condition.wetness ?? 0.0;
  const phase = condition.wavePhase ?? 0.0;
  const creaseAngle = condition.creaseAngle ?? 0.0;
  const creaseOffset = condition.creaseOffset ?? 0.0;
  const creaseStrength = Math.min(1.0, crinkle * 0.75 + (condition.creaseCount ?? 0) * 0.08);
  const creaseCos = Math.cos(creaseAngle);
  const creaseSin = Math.sin(creaseAngle);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const base = curl * x * x + ripple * Math.sin(x * 10.0) * Math.sin((y + 0.34) * 7.0);
    const wrinkles = crinkle * 0.0065 * (
      Math.sin(x * 28.0 + phase) * Math.sin(y * 17.0 - phase * 0.7)
      + 0.5 * Math.sin((x + y) * 36.0 + phase * 1.7)
    );
    const creaseDistance = x * creaseCos + y * creaseSin - creaseOffset;
    const crease = creaseStrength * 0.012 * Math.exp(-(creaseDistance * creaseDistance) / 0.0028);
    const dampSag = wetness * 0.0025 * Math.sin((x - y) * 12.0 + phase);
    const z = base + wrinkles + crease + dampSag;
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

function makeOccluderGeometry(occluder) {
  if (occluder.kind === "cover_card") {
    const geometry = new THREE.PlaneGeometry(occluder.width, occluder.height, 1, 1);
    geometry.computeVertexNormals();
    return geometry;
  }
  return makeFingerGeometry(occluder.radius, occluder.length);
}

function browserMulberry32(seed) {
  return () => {
    let value = seed += 0x6D2B79F5;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value || 0));
}

function applyMaskedOverlay(baseCanvas, drawOverlay) {
  const overlay = document.createElement("canvas");
  overlay.width = baseCanvas.width;
  overlay.height = baseCanvas.height;
  const overlayContext = overlay.getContext("2d");
  drawOverlay(overlayContext, overlay.width, overlay.height);
  overlayContext.globalCompositeOperation = "destination-in";
  overlayContext.drawImage(baseCanvas, 0, 0);
  return overlay;
}

function drawConditionedLines(context, width, height, rng, condition) {
  const crinkle = clamp01(condition.crinkle);
  const count = Math.max(0, condition.creaseCount || 0);
  for (let i = 0; i < count; i += 1) {
    const angle = (condition.creaseAngle || 0) + (rng() - 0.5) * 0.9;
    const offset = (rng() - 0.5) * height * 0.7;
    context.save();
    context.translate(width / 2, height / 2 + offset);
    context.rotate(angle);
    context.lineCap = "round";
    context.lineWidth = Math.max(1, height * (0.0025 + crinkle * 0.003));
    context.strokeStyle = "rgba(83,64,42," + (0.08 + crinkle * 0.14) + ")";
    context.beginPath();
    context.moveTo(-width * 0.62, 0);
    context.lineTo(width * 0.62, 0);
    context.stroke();
    context.strokeStyle = "rgba(255,248,226," + (0.05 + crinkle * 0.09) + ")";
    context.beginPath();
    context.moveTo(-width * 0.55, context.lineWidth * 1.3);
    context.lineTo(width * 0.55, context.lineWidth * 1.3);
    context.stroke();
    context.restore();
  }
}

function makeConditionedNoteTexture(sourceTexture, condition) {
  if (!condition) return sourceTexture;
  const dirt = clamp01(condition.dirtiness);
  const crinkle = clamp01(condition.crinkle);
  const wet = clamp01(condition.wetness);
  const edgeWear = clamp01(condition.edgeWear);
  if (dirt < 0.01 && crinkle < 0.01 && wet < 0.01 && edgeWear < 0.01) return sourceTexture;

  const source = sourceTexture.image;
  const width = source.naturalWidth || source.width;
  const height = source.naturalHeight || source.height;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.drawImage(source, 0, 0, width, height);
  const rng = browserMulberry32(condition.seed || 1);

  const wash = applyMaskedOverlay(canvas, (overlay, w, h) => {
    overlay.fillStyle = "rgba(117,82,38," + (dirt * 0.10) + ")";
    overlay.fillRect(0, 0, w, h);
    if (wet > 0) {
      overlay.fillStyle = "rgba(29,43,47," + (wet * 0.16) + ")";
      overlay.fillRect(0, 0, w, h);
    }
  });
  context.drawImage(wash, 0, 0);

  const marks = applyMaskedOverlay(canvas, (overlay, w, h) => {
    const speckles = Math.min(260, Math.max(0, condition.speckleCount || 0));
    for (let i = 0; i < speckles; i += 1) {
      const x = rng() * w;
      const y = rng() * h;
      const radius = Math.max(0.6, (0.003 + rng() * 0.011) * Math.min(w, h));
      overlay.fillStyle = "rgba(54,42,27," + (0.025 + dirt * (0.08 + rng() * 0.12)) + ")";
      overlay.beginPath();
      overlay.ellipse(x, y, radius * (0.6 + rng()), radius * (0.4 + rng() * 0.8), rng() * Math.PI, 0, Math.PI * 2);
      overlay.fill();
    }
    const stains = Math.min(8, Math.max(0, condition.stainCount || 0));
    for (let i = 0; i < stains; i += 1) {
      const x = rng() * w;
      const y = rng() * h;
      const radius = (0.04 + rng() * 0.11) * Math.min(w, h);
      const gradient = overlay.createRadialGradient(x, y, radius * 0.1, x, y, radius);
      gradient.addColorStop(0, "rgba(67,48,28," + (0.09 + dirt * 0.14) + ")");
      gradient.addColorStop(1, "rgba(67,48,28,0)");
      overlay.fillStyle = gradient;
      overlay.fillRect(x - radius, y - radius, radius * 2, radius * 2);
    }
    if (wet > 0) {
      const patches = Math.max(1, Math.ceil(wet * 5));
      for (let i = 0; i < patches; i += 1) {
        const x = rng() * w;
        const y = rng() * h;
        const radius = (0.07 + rng() * 0.15) * Math.min(w, h);
        const gradient = overlay.createRadialGradient(x, y, radius * 0.1, x, y, radius);
        gradient.addColorStop(0, "rgba(22,28,25," + (0.14 + wet * 0.20) + ")");
        gradient.addColorStop(0.55, "rgba(44,48,39," + (0.06 + wet * 0.10) + ")");
        gradient.addColorStop(1, "rgba(44,48,39,0)");
        overlay.fillStyle = gradient;
        overlay.fillRect(x - radius, y - radius, radius * 2, radius * 2);
      }
    }
    if (wet > 0.05) {
      const wetPatches = Math.min(6, Math.max(1, Math.round(1 + wet * 5)));
      for (let i = 0; i < wetPatches; i += 1) {
        const x = rng() * w;
        const y = rng() * h;
        const radiusX = (0.055 + rng() * 0.15) * w;
        const radiusY = (0.045 + rng() * 0.13) * h;
        const gradient = overlay.createRadialGradient(x, y, Math.min(radiusX, radiusY) * 0.08, x, y, Math.max(radiusX, radiusY));
        gradient.addColorStop(0, "rgba(18,29,31," + (wet * (0.10 + rng() * 0.08)) + ")");
        gradient.addColorStop(0.62, "rgba(46,64,64," + (wet * 0.07) + ")");
        gradient.addColorStop(1, "rgba(46,64,64,0)");
        overlay.save();
        overlay.translate(x, y);
        overlay.rotate(rng() * Math.PI);
        overlay.fillStyle = gradient;
        overlay.beginPath();
        overlay.ellipse(0, 0, radiusX, radiusY, 0, 0, Math.PI * 2);
        overlay.fill();
        overlay.restore();
      }
      const glints = Math.min(5, Math.max(1, Math.round(wet * 4)));
      overlay.save();
      overlay.globalCompositeOperation = "screen";
      for (let i = 0; i < glints; i += 1) {
        const x = rng() * w;
        const y = rng() * h;
        overlay.strokeStyle = "rgba(220,234,225," + (wet * (0.08 + rng() * 0.08)) + ")";
        overlay.lineWidth = Math.max(1, Math.min(w, h) * (0.002 + rng() * 0.004));
        overlay.lineCap = "round";
        overlay.beginPath();
        overlay.moveTo(x, y);
        overlay.bezierCurveTo(
          x + (rng() - 0.5) * w * 0.14,
          y + (rng() - 0.5) * h * 0.05,
          x + (rng() - 0.5) * w * 0.24,
          y + (rng() - 0.5) * h * 0.10,
          x + (rng() - 0.5) * w * 0.30,
          y + (rng() - 0.5) * h * 0.12
        );
        overlay.stroke();
      }
      overlay.restore();
    }
    drawConditionedLines(overlay, w, h, rng, condition);
    if (edgeWear > 0) {
      overlay.lineWidth = Math.max(1, Math.min(w, h) * (0.004 + edgeWear * 0.010));
      overlay.strokeStyle = "rgba(246,238,211," + (edgeWear * 0.18) + ")";
      for (let i = 0; i < 3; i += 1) {
        const inset = i * overlay.lineWidth * 1.8;
        overlay.strokeRect(inset, inset, w - inset * 2, h - inset * 2);
      }
    }
  });
  context.drawImage(marks, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

async function addNotes() {
  for (const asset of [...assets].sort((a, b) => a.layer - b.layer)) {
    const sourceTexture = await loader.loadAsync(asset.textureUrl);
    const texture = makeConditionedNoteTexture(sourceTexture, asset.condition);
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
    bendGeometry(geometry, asset.curl ?? 0.075, asset.ripple ?? 0.0, asset.condition);
    const condition = asset.condition || {};
    const materialRoughness = Math.max(
      0.24,
      Math.min(0.96, (asset.roughness ?? 0.82) + (condition.dirtiness || 0) * 0.04 - (condition.wetness || 0) * 0.52)
    );
    const backingMaterial = new THREE.MeshStandardMaterial({
      color: 0xf2ead7,
      roughness: Math.max(0.34, 0.92 - (condition.wetness || 0) * 0.32),
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
      roughness: materialRoughness,
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
    const geometry = makeOccluderGeometry(occluder);
    const material = new THREE.MeshStandardMaterial({
      color: occluder.color,
      roughness: 0.88,
      metalness: 0.0,
      side: THREE.DoubleSide,
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
    scene.environment = null;
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
    scene.environment = environmentTexture;
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
  const visualFilter = [
    "contrast(" + sceneConfig.postprocess.contrast + ")",
    "saturate(" + sceneConfig.postprocess.saturation + ")",
    "brightness(" + sceneConfig.postprocess.brightness + ")",
    sceneConfig.postprocess.focusBlurPx > 0 ? "blur(" + sceneConfig.postprocess.focusBlurPx + "px)" : "",
  ].filter(Boolean).join(" ");
  visualRenderer.domElement.style.filter = mode === "id" ? "none" : visualFilter;
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
  if (!fs.existsSync(BROWSER_EXECUTABLE)) {
    throw new Error(`Browser executable not found at ${BROWSER_EXECUTABLE}`);
  }
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const textureAssets = assets.map((asset) => ({ ...asset, textureUrl: pathToFileURL(asset.path).href }));
  const browser = await puppeteer.launch({
    executablePath: BROWSER_EXECUTABLE,
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
    const labelText = boxes
      .map((box) => `${box.classIndex} ${box.yolo.map((value) => Number(value).toFixed(6)).join(" ")}`)
      .join("\n");
    fs.writeFileSync(
      path.join(OUT_DIR, "labels_visible.txt"),
      labelText ? `${labelText}\n` : ""
    );
    fs.writeFileSync(
      path.join(OUT_DIR, "metadata.json"),
      JSON.stringify({
        renderer: "three-webgl-edge",
        browserExecutable: BROWSER_EXECUTABLE,
        variant: VARIANT,
        sceneMode: effectiveSceneMode,
        width: WIDTH,
        height: HEIGHT,
        visualScale: VISUAL_SCALE,
        minVisiblePixels: MIN_VISIBLE_PIXELS,
        sceneConfig: config,
        assetSelection: {
          sidePolicy: ASSET_SIDE_POLICY,
          classSequence: CLASS_SEQUENCE,
          sideCounts: assetSideCounts,
          frontBackMixSatisfied,
        },
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
