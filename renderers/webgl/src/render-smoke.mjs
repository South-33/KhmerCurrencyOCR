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
const ASSET_QUALITY_POLICY = argValue("--asset-quality-policy", "latest_design");
const STACK_POSE_POLICY = argValue("--stack-pose-policy", "default");
const CLEAN_ORIENTATION_POLICY = argValue("--clean-orientation-policy", "default");
const CAMERA_PROFILE = argValue("--camera-profile", "generic_phone_jitter");
const CLASS_SEQUENCE_RAW = argValue("--class-sequence", "");
const NOTE_CONDITION_POLICY = argValue("--note-condition-policy", "mixed");
const LENS_DISTORTION_POLICY = argValue("--lens-distortion-policy", "off");
const NOTE_PRINT_TONE_POLICY = argValue("--note-print-tone-policy", "off");
const CAMERA_ISP_POLICY = argValue("--camera-isp-policy", "default");
const TEXTURE_QA_EFFECTS = argValue("--texture-qa-effects", "flat");
const APPEARANCE_ABLATION = argValue("--appearance-ablation", "full");
const OCCLUDER_POLICY = argValue("--occluder-policy", "scene_default");
const NEGATIVE_PROP_POLICY = argValue("--negative-prop-policy", "classic");
const NEGATIVE_PROP_LABEL_POLICY = argValue("--negative-prop-label-policy", "none");
const BROWSER_EXECUTABLE = argValue("--browser-executable", process.env.CASHSNAP_WEBGL_BROWSER || EDGE);
const BROWSER_WS_ENDPOINT = argValue("--browser-ws-endpoint", process.env.CASHSNAP_WEBGL_BROWSER_WS || "");
const BATCH_MANIFEST = argValue("--batch-manifest", "");
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
const UNKNOWN_PROP_CLASS_NAME = "UNKNOWN_FOREIGN_NOTE";
const UNKNOWN_PROP_CLASS_INDEX = CLASS_NAMES.length;
const UNKNOWN_PROP_ID_CHANNELS = [64, 96, 128, 160, 192, 224, 255];
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

if (!["auto", "clean", "clean_single", "clean_context", "texture_qa", "negative", "stack", "fan", "thin_edge", "hand_occlusion", "qa3"].includes(SCENE_MODE)) {
  throw new Error("--scene-mode must be one of: auto, clean, clean_single, clean_context, texture_qa, negative, stack, fan, thin_edge, hand_occlusion, qa3");
}

if (!["any", "front_only", "back_only", "front_back_mix"].includes(ASSET_SIDE_POLICY)) {
  throw new Error("--asset-side-policy must be one of: any, front_only, back_only, front_back_mix");
}

if (!["latest_design", "all_manifest", "filesystem_all"].includes(ASSET_QUALITY_POLICY)) {
  throw new Error("--asset-quality-policy must be one of: latest_design, all_manifest, filesystem_all");
}

if (!["default", "real_aspect_v1", "real_aspect_v2"].includes(STACK_POSE_POLICY)) {
  throw new Error("--stack-pose-policy must be one of: default, real_aspect_v1, real_aspect_v2");
}

if (!["default", "real_aspect_v1", "real_aspect_square_v1", "real_aspect_bridge_v1"].includes(CLEAN_ORIENTATION_POLICY)) {
  throw new Error("--clean-orientation-policy must be one of: default, real_aspect_v1, real_aspect_square_v1, real_aspect_bridge_v1");
}

if (![
  "generic_phone_jitter",
  "phone_auto",
  "iphone_8_like",
  "iphone_12_wide_like",
  "budget_android_wide_like",
  "browser_upload_resized",
  "phone_closeup_clean_like",
  "phone_clean_base_readable_mix_v1",
  "phone_clean_base_topdown_readable_v1",
  "phone_clean_base_square_topdown_readable_v1",
  "phone_bridge_square_topdown_v1",
  "phone_top_down_like",
  "phone_oblique_30_like",
  "phone_oblique_45_like",
  "phone_low_front_like",
].includes(CAMERA_PROFILE)) {
  throw new Error("--camera-profile must be one of: generic_phone_jitter, phone_auto, iphone_8_like, iphone_12_wide_like, budget_android_wide_like, browser_upload_resized, phone_closeup_clean_like, phone_clean_base_readable_mix_v1, phone_clean_base_topdown_readable_v1, phone_clean_base_square_topdown_readable_v1, phone_bridge_square_topdown_v1, phone_top_down_like, phone_oblique_30_like, phone_oblique_45_like, phone_low_front_like");
}

if (!["mixed", "scan_fidelity", "pristine_only", "handled_clean", "handled_3d", "heavy_wear", "wet_stress"].includes(NOTE_CONDITION_POLICY)) {
  throw new Error("--note-condition-policy must be one of: mixed, scan_fidelity, pristine_only, handled_clean, handled_3d, heavy_wear, wet_stress");
}

if (!["off", "phone_mild"].includes(LENS_DISTORTION_POLICY)) {
  throw new Error("--lens-distortion-policy must be one of: off, phone_mild");
}

if (!["off", "local_dynamic_range_v1", "bill_auto_exposure_v1", "real_bridge_print_contrast_v1"].includes(NOTE_PRINT_TONE_POLICY)) {
  throw new Error("--note-print-tone-policy must be one of: off, local_dynamic_range_v1, bill_auto_exposure_v1, real_bridge_print_contrast_v1");
}

if (!["default", "phone_dynamic_range_v1", "phone_dynamic_range_v2", "real_bridge_dynamic_range_v1", "mined_fp_dark_v1"].includes(CAMERA_ISP_POLICY)) {
  throw new Error("--camera-isp-policy must be one of: default, phone_dynamic_range_v1, phone_dynamic_range_v2, real_bridge_dynamic_range_v1, mined_fp_dark_v1");
}

if (!["flat", "lit_material", "backing_plane", "postprocess", "condition"].includes(TEXTURE_QA_EFFECTS)) {
  throw new Error("--texture-qa-effects must be one of: flat, lit_material, backing_plane, postprocess, condition");
}

if (!["full", "source_flat", "material", "material_no_shadow", "material_no_backing", "standard_only", "texture_resolve", "postprocess", "grain", "condition"].includes(APPEARANCE_ABLATION)) {
  throw new Error("--appearance-ablation must be one of: full, source_flat, material, material_no_shadow, material_no_backing, standard_only, texture_resolve, postprocess, grain, condition");
}

if (!["scene_default", "no_hand", "none"].includes(OCCLUDER_POLICY)) {
  throw new Error("--occluder-policy must be one of: scene_default, no_hand, none");
}

if (!["none", "unknown_banknote"].includes(NEGATIVE_PROP_LABEL_POLICY)) {
  throw new Error("--negative-prop-label-policy must be one of: none, unknown_banknote");
}

if (!["classic", "unknown_currency_soft_v1", "unknown_currency_soft_dark_v1", "unknown_currency_spread_dark_v1", "unknown_currency_v1", "unknown_currency_fullframe_v1", "unknown_currency_fullframe_dark_v1"].includes(NEGATIVE_PROP_POLICY)) {
  throw new Error("--negative-prop-policy must be one of: classic, unknown_currency_soft_v1, unknown_currency_soft_dark_v1, unknown_currency_spread_dark_v1, unknown_currency_v1, unknown_currency_fullframe_v1, unknown_currency_fullframe_dark_v1");
}

const ABLATION_ACTIVE = APPEARANCE_ABLATION !== "full";
const ABLATION_MATERIAL_SPLIT = ["material", "material_no_shadow", "material_no_backing", "standard_only"].includes(APPEARANCE_ABLATION);
const ABLATION_DISABLE_CONDITION = ["source_flat", "texture_resolve", "postprocess", "grain"].includes(APPEARANCE_ABLATION) || ABLATION_MATERIAL_SPLIT;
const ABLATION_SOURCE_EXACT = APPEARANCE_ABLATION === "source_flat" || ABLATION_MATERIAL_SPLIT;
const ABLATION_DISABLE_POSTPROCESS = ["source_flat", "texture_resolve", "condition"].includes(APPEARANCE_ABLATION) || ABLATION_MATERIAL_SPLIT;
const ABLATION_DISABLE_GRAIN = ["source_flat", "texture_resolve", "postprocess", "condition"].includes(APPEARANCE_ABLATION) || ABLATION_MATERIAL_SPLIT;
const ABLATION_FLAT_MATERIAL = APPEARANCE_ABLATION === "source_flat";

function effectiveSceneModeFor(variant, sceneMode) {
  return sceneMode === "auto" ? (variant >= 100 ? "fan" : "stack") : sceneMode;
}

let effectiveSceneMode = effectiveSceneModeFor(VARIANT, SCENE_MODE);

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

function randomSignedBetween(rng, minAbs, maxAbs) {
  const magnitude = randomBetween(rng, minAbs, maxAbs);
  return rng() < 0.5 ? -magnitude : magnitude;
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
  if (mode === "texture_qa") return 1;
  if (mode === "clean_single") return 1;
  if (mode === "clean_context") return 1;
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
  const textureQaScene = effectiveSceneMode === "texture_qa";
  const forceTextureQaScan = textureQaScene && !["condition"].includes(TEXTURE_QA_EFFECTS);
  const forceScanFidelity = forceTextureQaScan || ABLATION_DISABLE_CONDITION;
  const cleanScene = Boolean(asset.clean || asset.cleanSingle || asset.cleanContext || textureQaScene || effectiveSceneMode === "clean" || effectiveSceneMode === "clean_single" || effectiveSceneMode === "clean_context");
  let dirtiness = 0;
  let crinkle = 0;
  let wetness = 0;
  let textureCreaseMarks = true;
  let geometryCurlScale = 1.0;
  let geometryRippleScale = 1.0;
  let geometryWrinkleScale = 1.0;
  let geometryCreaseScale = 1.0;
  let geometryCreaseWidth = 0.0028;
  let creaseCountOverride = null;
  let edgeWearOverride = null;
  let stainCountOverride = null;
  let speckleCountOverride = null;
  if (NOTE_CONDITION_POLICY === "scan_fidelity" || forceScanFidelity) {
    return {
      profile: "scan_fidelity",
      policy: ABLATION_DISABLE_CONDITION ? "scan_fidelity_ablation" : NOTE_CONDITION_POLICY,
      seed,
      dirtiness: 0,
      crinkle: 0,
      wetness: 0,
      edgeWear: 0,
      creaseCount: 0,
      stainCount: 0,
      speckleCount: 0,
      creaseAngle: 0,
      creaseOffset: 0,
      wavePhase: 0,
      textureCreaseMarks: false,
      geometryCurlScale: 1.0,
      geometryRippleScale: 1.0,
      geometryWrinkleScale: 0.0,
      geometryCreaseScale: 0.0,
      geometryCreaseWidth: 0.0028,
    };
  }
  if (NOTE_CONDITION_POLICY === "pristine_only") {
    dirtiness = randomBetween(rng, 0.00, 0.08);
    crinkle = randomBetween(rng, 0.00, 0.14);
  } else if (NOTE_CONDITION_POLICY === "handled_clean") {
    const bucket = rng();
    if (cleanScene) {
      if (bucket < 0.72) {
        dirtiness = randomBetween(rng, 0.00, 0.08);
        crinkle = randomBetween(rng, 0.00, 0.06);
      } else {
        dirtiness = randomBetween(rng, 0.06, 0.18);
        crinkle = randomBetween(rng, 0.04, 0.14);
      }
    } else if (bucket < 0.14) {
      dirtiness = randomBetween(rng, 0.04, 0.16);
      crinkle = randomBetween(rng, 0.02, 0.18);
    } else if (bucket < 0.78) {
      dirtiness = randomBetween(rng, 0.20, 0.50);
      crinkle = randomBetween(rng, 0.12, 0.46);
    } else {
      dirtiness = randomBetween(rng, 0.50, 0.78);
      crinkle = randomBetween(rng, 0.34, 0.70);
    }
    wetness = !cleanScene && rng() < 0.05 ? randomBetween(rng, 0.04, 0.18) : 0.0;
  } else if (NOTE_CONDITION_POLICY === "handled_3d") {
    const bucket = rng();
    if (cleanScene) {
      dirtiness = bucket < 0.62
        ? randomBetween(rng, 0.04, 0.14)
        : randomBetween(rng, 0.12, 0.24);
      crinkle = bucket < 0.56
        ? randomBetween(rng, 0.26, 0.50)
        : randomBetween(rng, 0.48, 0.72);
    } else if (bucket < 0.72) {
      dirtiness = randomBetween(rng, 0.12, 0.36);
      crinkle = randomBetween(rng, 0.32, 0.66);
    } else {
      dirtiness = randomBetween(rng, 0.36, 0.68);
      crinkle = randomBetween(rng, 0.52, 0.88);
    }
    wetness = 0.0;
    textureCreaseMarks = false;
    geometryCurlScale = randomBetween(rng, 1.80, 3.40);
    geometryRippleScale = randomBetween(rng, 2.20, 4.30);
    geometryWrinkleScale = randomBetween(rng, 3.00, 6.50);
    geometryCreaseScale = randomBetween(rng, 5.50, 10.50);
    geometryCreaseWidth = randomBetween(rng, 0.0010, 0.0024);
    creaseCountOverride = 2 + randomInt(rng, cleanScene ? 3 : 4);
    edgeWearOverride = cleanScene
      ? 0
      : clamp(dirtiness * randomBetween(rng, 0.18, 0.55) + crinkle * 0.12, 0, 0.58);
    stainCountOverride = Math.max(0, Math.round(dirtiness * randomBetween(rng, 1.0, 3.2)));
    speckleCountOverride = Math.max(8, Math.round(14 + dirtiness * randomBetween(rng, 48, 120)));
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
  const edgeWear = edgeWearOverride === null ? (cleanScene ? 0 : clamp(dirtiness * randomBetween(rng, 0.40, 1.10) + crinkle * 0.20, 0, 1)) : edgeWearOverride;
  const creaseCount = creaseCountOverride === null ? (cleanScene ? 0 : Math.max(0, Math.round(crinkle * randomBetween(rng, 1.0, 5.0)))) : creaseCountOverride;
  const stainCount = stainCountOverride === null ? (cleanScene ? 0 : Math.max(0, Math.round(dirtiness * randomBetween(rng, 1.4, 6.2) + wetness * 3.0))) : stainCountOverride;
  const speckleCount = speckleCountOverride === null ? (cleanScene ? Math.max(0, Math.round(dirtiness * randomBetween(rng, 8, 36))) : Math.max(0, Math.round(12 + dirtiness * randomBetween(rng, 110, 270)))) : speckleCountOverride;
  const profile = wetness > 0.42
    ? "damp"
    : NOTE_CONDITION_POLICY === "handled_3d"
      ? crinkle >= 0.58
        ? "bent_handled_strong"
        : "bent_handled"
    : NOTE_CONDITION_POLICY === "handled_clean"
      ? dirtiness >= 0.52 || crinkle >= 0.40
        ? "well_handled"
        : dirtiness < 0.18 && crinkle < 0.20
          ? "lightly_handled"
          : "circulated"
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
    textureCreaseMarks,
    geometryCurlScale: round3(geometryCurlScale),
    geometryRippleScale: round3(geometryRippleScale),
    geometryWrinkleScale: round3(geometryWrinkleScale),
    geometryCreaseScale: round3(geometryCreaseScale),
    geometryCreaseWidth: round3(geometryCreaseWidth),
  };
}

function notePrintToneFor(asset, variant, index, condition = {}) {
  const seed = 90714113 + variant * 1297 + index * 3253 + (asset.classIndex ?? 0) * 101;
  if (NOTE_PRINT_TONE_POLICY === "off") {
    return {
      policy: "off",
      seed,
      contrast: 1.0,
      saturation: 1.0,
      brightness: 0.0,
      shadowPull: 0.0,
      highlightPush: 0.0,
    };
  }
  const rng = mulberry32(seed);
  const wetness = clamp(condition.wetness || 0, 0, 1);
  const dirtiness = clamp(condition.dirtiness || 0, 0, 1);
  const crinkle = clamp(condition.crinkle || 0, 0, 1);
  const cleanScene = Boolean(asset.clean || asset.cleanSingle || asset.cleanContext || effectiveSceneMode === "clean" || effectiveSceneMode === "clean_single" || effectiveSceneMode === "clean_context");
  const wetDampening = 1.0 - wetness * 0.13;
  const wearBoost = 1.0 + Math.min(0.12, (dirtiness + crinkle) * 0.055);
  const cleanDampening = cleanScene ? 0.92 : 1.0;
  if (NOTE_PRINT_TONE_POLICY === "bill_auto_exposure_v1") {
    return {
      policy: NOTE_PRINT_TONE_POLICY,
      seed,
      contrast: round3(randomBetween(rng, 1.22, 1.58) * wetDampening * wearBoost * cleanDampening),
      saturation: round3(randomBetween(rng, 0.92, 1.08)),
      brightness: round3(randomBetween(rng, -16.0, -3.0) - dirtiness * 2.0),
      shadowPull: round3(randomBetween(rng, 10.0, 34.0) * (1.0 + dirtiness * 0.18)),
      highlightPush: round3(randomBetween(rng, -24.0, -5.0) * (1.0 - wetness * 0.20)),
    };
  }
  if (NOTE_PRINT_TONE_POLICY === "real_bridge_print_contrast_v1") {
    return {
      policy: NOTE_PRINT_TONE_POLICY,
      seed,
      contrast: round3(randomBetween(rng, 1.80, 2.45) * wetDampening * wearBoost * cleanDampening),
      saturation: round3(randomBetween(rng, 0.84, 1.24)),
      brightness: round3(randomBetween(rng, -28.0, -7.0) - dirtiness * 2.5),
      shadowPull: round3(randomBetween(rng, 42.0, 88.0) * (1.0 + dirtiness * 0.24)),
      highlightPush: round3(randomBetween(rng, 10.0, 34.0) * (1.0 - wetness * 0.18)),
    };
  }
  const contrast = randomBetween(rng, 1.52, 1.92) * wetDampening * wearBoost * cleanDampening;
  return {
    policy: NOTE_PRINT_TONE_POLICY,
    seed,
    contrast: round3(contrast),
    saturation: round3(randomBetween(rng, 1.02, 1.16)),
    brightness: round3(randomBetween(rng, -18.0, -4.0) - dirtiness * 3.0),
    shadowPull: round3(randomBetween(rng, 24.0, 58.0) * (1.0 + dirtiness * 0.25)),
    highlightPush: round3(randomBetween(rng, 8.0, 22.0) * (1.0 - wetness * 0.20)),
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
  phone_clean_base_readable_mix_v1: {
    source: "cashsnap_clean_base_visual_qa_v1",
    weight: 0.0,
    targetResolution: [1440, 1080],
    fov: [36, 62],
    positionX: [-0.14, 0.14],
    positionY: [-0.60, -0.02],
    positionZ: [1.15, 1.95],
    lookAtX: [-0.06, 0.06],
    lookAtY: [-0.04, 0.10],
  },
  phone_clean_base_topdown_readable_v1: {
    source: "cashsnap_clean_base_visual_qa_v2",
    weight: 0.0,
    targetResolution: [1440, 1080],
    fov: [38, 56],
    positionX: [-0.10, 0.10],
    positionY: [-0.18, 0.08],
    positionZ: [1.30, 1.75],
    lookAtX: [-0.04, 0.04],
    lookAtY: [-0.04, 0.06],
  },
  phone_clean_base_square_topdown_readable_v1: {
    source: "cashsnap_square_topdown_geometry_bracket_v1",
    weight: 0.0,
    targetResolution: [640, 640],
    fov: [40, 58],
    positionX: [-0.09, 0.09],
    positionY: [-0.14, 0.06],
    positionZ: [1.50, 1.95],
    lookAtX: [-0.04, 0.04],
    lookAtY: [-0.04, 0.06],
  },
  phone_bridge_square_topdown_v1: {
    source: "roboflow_core13_bridge_geometry_v1",
    weight: 0.0,
    targetResolution: [640, 640],
    fov: [50, 68],
    positionX: [-0.12, 0.12],
    positionY: [-0.18, 0.10],
    positionZ: [1.72, 2.30],
    lookAtX: [-0.05, 0.05],
    lookAtY: [-0.05, 0.07],
  },
  texture_qa_flat: {
    source: "cashsnap_texture_fidelity_review_v1",
    weight: 0.0,
    targetResolution: [1440, 1080],
    fov: [36, 36],
    positionX: [0.0, 0.0],
    positionY: [0.0, 0.0],
    positionZ: [1.48, 1.48],
    lookAtX: [0.0, 0.0],
    lookAtY: [0.0, 0.0],
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

function lensDistortionConfig(rng, cameraProfile) {
  if (LENS_DISTORTION_POLICY === "off") {
    return {
      policy: "off",
      applied: false,
      model: "none",
      k1: 0,
    };
  }
  const wideProfile = /wide|low_front|oblique_45/.test(cameraProfile.name);
  const mostlyBarrel = rng() < 0.85;
  const k1 = mostlyBarrel
    ? randomBetween(rng, wideProfile ? -0.060 : -0.040, wideProfile ? -0.020 : -0.010)
    : randomBetween(rng, 0.006, wideProfile ? 0.018 : 0.014);
  return {
    policy: LENS_DISTORTION_POLICY,
    applied: true,
    model: "radial_k1_inverse_remap",
    k1: round3(k1),
    centerOffset: [round3(randomBetween(rng, -0.018, 0.018)), round3(randomBetween(rng, -0.018, 0.018))],
    frameScale: round3(k1 < 0 ? Math.max(0.86, 1 + k1 * 1.8) : 1.0),
    visualSampling: "bilinear",
    idSampling: "nearest",
    sharedRgbIdLabelTransform: true,
    labelsDerivedFrom: "postprocessed_id_pixels",
  };
}

function sceneConfig(variant, mode, backgroundPath, environmentPath) {
  const modeOffset = mode === "fan" ? 1009 : mode === "clean" ? 2003 : mode === "clean_single" ? 2309 : mode === "clean_context" ? 2609 : mode === "texture_qa" ? 2801 : mode === "qa3" ? 3001 : mode === "negative" ? 4001 : mode === "thin_edge" ? 5003 : mode === "hand_occlusion" ? 6007 : 0;
  const rng = mulberry32(26058003 + variant * 191 + modeOffset);
  const textureQa = mode === "texture_qa";
  const cameraProfile = textureQa ? { name: "texture_qa_flat", ...CAMERA_PROFILES.texture_qa_flat } : chooseCameraProfile(rng, CAMERA_PROFILE);
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
  const cleanContextSurfaces = [
    ...cleanSingleSurfaces,
    { name: "dark_shop_counter", base: "#4f473f", light: "#aa9a84", dark: "#1f1b17", scene: "#554c43", repeat: [2.4, 1.9] },
    { name: "red_brown_laminate", base: "#795a4e", light: "#c59b86", dark: "#241816", scene: "#80665c", repeat: [2.6, 2.0] },
  ];
  const textureQaSurfaces = [
    { name: "texture_qa_flat_gray", base: "#d8d8d8", light: "#d8d8d8", dark: "#d8d8d8", scene: "#d8d8d8", repeat: [1.0, 1.0], flatTexture: true },
  ];
  const minedFpDarkSurfaces = [
    { name: "dim_black_counter", base: "#16120f", light: "#4a382d", dark: "#050403", scene: "#15110f", repeat: [2.2, 1.8] },
    { name: "deep_blue_phone_shadow", base: "#111722", light: "#35475c", dark: "#020407", scene: "#0f1520", repeat: [1.9, 1.6] },
    { name: "dark_red_brown_laminate", base: "#29150f", light: "#5f3426", dark: "#070302", scene: "#24120d", repeat: [2.5, 1.9] },
    { name: "low_light_gray_pad", base: "#1c1d1b", light: "#565852", dark: "#050604", scene: "#1a1b19", repeat: [2.0, 1.7] },
  ];
  const minedFpDarkNegative = mode === "negative" && (
    NEGATIVE_PROP_POLICY === "unknown_currency_fullframe_dark_v1"
    || NEGATIVE_PROP_POLICY === "unknown_currency_soft_dark_v1"
    || NEGATIVE_PROP_POLICY === "unknown_currency_spread_dark_v1"
  );
  const surfaces = textureQa
    ? textureQaSurfaces
    : minedFpDarkNegative
      ? minedFpDarkSurfaces
      : mode === "clean_single"
        ? cleanSingleSurfaces
        : mode === "clean_context"
          ? cleanContextSurfaces
          : defaultSurfaces;
  const surface = surfaces[randomInt(rng, surfaces.length)];
  const cleanSingle = mode === "clean_single";
  const cleanContext = mode === "clean_context";
  const cleanReadable = cleanSingle || cleanContext || textureQa;
  const textureQaFlat = textureQa && TEXTURE_QA_EFFECTS === "flat";
  const textureQaPhysicalCondition = textureQa && TEXTURE_QA_EFFECTS === "condition" && NOTE_CONDITION_POLICY === "handled_3d";
  const textureQaPostprocess = textureQa && TEXTURE_QA_EFFECTS === "postprocess";
  const textureResolve = textureQa || ABLATION_SOURCE_EXACT ? {
    policy: "source_exact",
    maxWidth: 0,
    blurPx: 0.0,
  } : cleanReadable ? {
    policy: "camera_lowpass_clean_v2",
    maxWidth: 1400,
    blurPx: 0.15,
  } : {
    policy: "camera_lowpass_scene_v2",
    maxWidth: 1600,
    blurPx: 0.10,
  };
  const environmentExtension = environmentPath ? path.extname(environmentPath).toLowerCase() : "";
  const lensDistortion = lensDistortionConfig(rng, cameraProfile);
  const postprocess = {
    contrast: textureQa && !textureQaPostprocess ? 1.0 : cleanReadable ? randomBetween(rng, 1.04, 1.14) : randomBetween(rng, 1.00, 1.07),
    saturation: textureQa && !textureQaPostprocess ? 1.0 : cleanSingle || textureQa ? randomBetween(rng, 1.10, 1.24) : cleanContext ? randomBetween(rng, 1.06, 1.30) : randomBetween(rng, 0.90, 1.07),
    brightness: textureQa && !textureQaPostprocess ? 1.0 : cleanSingle || textureQa ? randomBetween(rng, 0.94, 1.02) : cleanContext ? randomBetween(rng, 0.90, 1.03) : randomBetween(rng, 0.96, 1.05),
    focusBlurPx: textureQa && !textureQaPostprocess ? 0 : rng() < (cleanReadable ? (cleanContext || textureQa ? 0.30 : 0.20) : 0.68) ? randomBetween(rng, cleanReadable ? 0.01 : 0.05, cleanReadable ? 0.24 : 0.32) : 0,
    grainStrength: textureQa && !textureQaPostprocess ? 0 : randomBetween(rng, cleanReadable ? 3 : 12, cleanReadable ? 16 : 30),
    grainAlpha: textureQa && !textureQaPostprocess ? 0 : randomBetween(rng, cleanReadable ? 4 : 8, cleanReadable ? 14 : 18),
    vignette: textureQa && !textureQaPostprocess ? 0 : randomBetween(rng, cleanReadable ? 22 : 42, cleanReadable ? 58 : 78),
    grainSeed: Math.floor(rng() * 0x100000000),
  };
  if (CAMERA_ISP_POLICY === "phone_dynamic_range_v1" && !(textureQa && !textureQaPostprocess)) {
    postprocess.contrast = cleanReadable ? randomBetween(rng, 1.55, 2.20) : randomBetween(rng, 1.35, 1.95);
    postprocess.saturation = cleanReadable ? randomBetween(rng, 0.78, 1.46) : randomBetween(rng, 0.75, 1.35);
    postprocess.brightness = cleanReadable ? randomBetween(rng, 0.82, 1.06) : randomBetween(rng, 0.80, 1.08);
    postprocess.focusBlurPx = rng() < (cleanReadable ? 0.16 : 0.36) ? randomBetween(rng, 0.01, cleanReadable ? 0.12 : 0.24) : 0;
    postprocess.grainStrength = randomBetween(rng, cleanReadable ? 12 : 18, cleanReadable ? 34 : 42);
    postprocess.grainAlpha = randomBetween(rng, cleanReadable ? 10 : 12, cleanReadable ? 24 : 28);
    postprocess.vignette = randomBetween(rng, cleanReadable ? 36 : 50, cleanReadable ? 96 : 118);
  }
  if (CAMERA_ISP_POLICY === "phone_dynamic_range_v2" && !(textureQa && !textureQaPostprocess)) {
    postprocess.contrast = cleanReadable ? randomBetween(rng, 1.42, 2.02) : randomBetween(rng, 1.30, 1.85);
    postprocess.saturation = cleanReadable ? randomBetween(rng, 0.46, 0.90) : randomBetween(rng, 0.48, 0.96);
    postprocess.brightness = cleanReadable ? randomBetween(rng, 0.88, 1.10) : randomBetween(rng, 0.86, 1.10);
    postprocess.focusBlurPx = rng() < (cleanReadable ? 0.14 : 0.32) ? randomBetween(rng, 0.01, cleanReadable ? 0.10 : 0.22) : 0;
    postprocess.grainStrength = randomBetween(rng, cleanReadable ? 10 : 16, cleanReadable ? 30 : 38);
    postprocess.grainAlpha = randomBetween(rng, cleanReadable ? 9 : 11, cleanReadable ? 21 : 25);
    postprocess.vignette = randomBetween(rng, cleanReadable ? 30 : 46, cleanReadable ? 86 : 108);
  }
  if (CAMERA_ISP_POLICY === "real_bridge_dynamic_range_v1" && !(textureQa && !textureQaPostprocess)) {
    postprocess.contrast = cleanReadable ? randomBetween(rng, 1.95, 2.85) : randomBetween(rng, 1.70, 2.45);
    postprocess.saturation = cleanReadable ? randomBetween(rng, 0.72, 1.30) : randomBetween(rng, 0.70, 1.24);
    postprocess.brightness = cleanReadable ? randomBetween(rng, 0.76, 1.00) : randomBetween(rng, 0.74, 1.02);
    postprocess.focusBlurPx = rng() < (cleanReadable ? 0.20 : 0.38) ? randomBetween(rng, 0.01, cleanReadable ? 0.14 : 0.24) : 0;
    postprocess.grainStrength = randomBetween(rng, cleanReadable ? 14 : 20, cleanReadable ? 38 : 48);
    postprocess.grainAlpha = randomBetween(rng, cleanReadable ? 11 : 13, cleanReadable ? 27 : 32);
    postprocess.vignette = randomBetween(rng, cleanReadable ? 42 : 56, cleanReadable ? 110 : 130);
  }
  if (CAMERA_ISP_POLICY === "mined_fp_dark_v1" && !(textureQa && !textureQaPostprocess)) {
    postprocess.contrast = randomBetween(rng, 1.42, 2.18);
    postprocess.saturation = randomBetween(rng, 1.85, 3.10);
    postprocess.brightness = randomBetween(rng, 0.44, 0.74);
    postprocess.focusBlurPx = rng() < 0.24 ? randomBetween(rng, 0.02, 0.18) : 0;
    postprocess.grainStrength = randomBetween(rng, 24, 52);
    postprocess.grainAlpha = randomBetween(rng, 18, 34);
    postprocess.vignette = randomBetween(rng, 92, 160);
  }
  if (ABLATION_DISABLE_POSTPROCESS) {
    postprocess.contrast = 1.0;
    postprocess.saturation = 1.0;
    postprocess.brightness = 1.0;
    postprocess.focusBlurPx = 0;
    postprocess.vignette = 0;
  }
  if (ABLATION_DISABLE_GRAIN) {
    postprocess.grainStrength = 0;
    postprocess.grainAlpha = 0;
  }
  return {
    noteConditionPolicy: NOTE_CONDITION_POLICY,
    notePrintTonePolicy: NOTE_PRINT_TONE_POLICY,
    cameraIspPolicy: CAMERA_ISP_POLICY,
    assetQualityPolicy: ASSET_QUALITY_POLICY,
    cleanOrientationPolicy: CLEAN_ORIENTATION_POLICY,
    negativePropPolicy: NEGATIVE_PROP_POLICY,
    negativePropLabelPolicy: NEGATIVE_PROP_LABEL_POLICY,
    unknownPropClassName: UNKNOWN_PROP_CLASS_NAME,
    unknownPropClassIndex: UNKNOWN_PROP_CLASS_INDEX,
    appearanceAblation: APPEARANCE_ABLATION,
    textureQa,
    textureQaEffects: textureQa ? TEXTURE_QA_EFFECTS : "none",
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
      lensDistortion,
    },
    lighting: {
      hemiIntensity: textureQaFlat ? 0.0 : textureQaPhysicalCondition ? 1.25 : textureQa ? 2.1 : randomBetween(rng, 1.15, 1.95),
      keyIntensity: textureQaFlat ? 0.0 : textureQaPhysicalCondition ? 3.4 : textureQa ? 2.2 : randomBetween(rng, 1.35, 2.55),
      keyColor: textureQa ? 0xffffff : [0xffd8aa, 0xffffff, 0xdff1ff, 0xffefc8][randomInt(rng, 4)],
      keyPosition: textureQaPhysicalCondition ? [-1.9, -2.4, 3.2] : textureQa ? [0.0, 0.0, 4.2] : [
        randomBetween(rng, -2.2, 2.2),
        randomBetween(rng, -3.0, -1.2),
        randomBetween(rng, 2.4, 4.4),
      ],
    },
    shadowPolicy: {
      engineShadowMap: "vsm_4096_bounded_v1",
      noteContactShadow: "off",
      noteMeshesCastEngineShadow: true,
      noteMeshesReceiveEngineShadow: false,
      backingMeshesReceiveEngineShadow: false,
    },
    postprocess,
    textureResolve,
    labelTransformPolicy: lensDistortion.applied ? {
      geometricPostprocess: "shared_rgb_id_label_radial_warp",
      sharedGeometricPostprocess: true,
      labelsDerivedFrom: "postprocessed_id_pixels",
      visualSampling: lensDistortion.visualSampling,
      idSampling: lensDistortion.idSampling,
    } : {
      geometricPostprocess: "none",
      sharedGeometricPostprocess: false,
      labelsDerivedFrom: "raw_id_pixels",
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

const STACK_REAL_ASPECT_Z_ABS_RANGES = {
  USD_1: [0.10, 0.30],
  USD_5: [0.00, 0.12],
  USD_10: [0.76, 1.00],
  USD_20: [0.70, 0.94],
  USD_50: [0.05, 0.22],
  USD_100: [0.05, 0.22],
  KHR_500: [0.20, 0.42],
  KHR_1000: [0.32, 0.56],
  KHR_2000: [0.12, 0.32],
  KHR_5000: [0.38, 0.62],
  KHR_10000: [0.00, 0.10],
  KHR_20000: [0.02, 0.18],
  KHR_50000: [0.06, 0.24],
};

const STACK_REAL_ASPECT_V2_Z_ABS_RANGES = {
  ...STACK_REAL_ASPECT_Z_ABS_RANGES,
  USD_50: [0.00, 0.12],
  KHR_10000: [0.00, 0.06],
  KHR_20000: [0.00, 0.08],
};

const CLEAN_REAL_ASPECT_SQUARE_Z_ABS_RANGES = {
  ...STACK_REAL_ASPECT_Z_ABS_RANGES,
  USD_1: [0.16, 0.36],
  USD_5: [0.16, 0.34],
  USD_10: [0.84, 1.08],
  USD_50: [0.22, 0.42],
  USD_100: [0.22, 0.42],
  KHR_500: [0.30, 0.52],
};

const CLEAN_REAL_ASPECT_BRIDGE_Z_ABS_RANGES = {
  ...CLEAN_REAL_ASPECT_SQUARE_Z_ABS_RANGES,
  USD_10: [0.38, 0.66],
  USD_50: [0.18, 0.38],
  USD_100: [0.34, 0.58],
  KHR_500: [0.04, 0.22],
  KHR_2000: [0.34, 0.60],
  KHR_5000: [0.30, 0.52],
  KHR_50000: [0.00, 0.10],
};

const STACK_REAL_ASPECT_V2_POSITION_LIMITS = {
  USD_50: { x: [-0.28, 0.18], y: [-0.12, 0.14] },
  KHR_10000: { x: [-0.30, 0.22], y: [-0.12, 0.18] },
  KHR_20000: { x: [-0.32, 0.22], y: [-0.12, 0.18] },
};

function stackRealAspectZRanges() {
  if (STACK_POSE_POLICY === "real_aspect_v2") return STACK_REAL_ASPECT_V2_Z_ABS_RANGES;
  return STACK_REAL_ASPECT_Z_ABS_RANGES;
}

function stackZRotationForClass(className, baseZ, rng) {
  if (STACK_POSE_POLICY === "default") {
    return baseZ + randomBetween(rng, -0.34, 0.34);
  }
  const range = stackRealAspectZRanges()[className];
  if (!range) {
    throw new Error(`No stack real-aspect z-rotation range for class ${className}`);
  }
  return randomSignedBetween(rng, range[0], range[1]);
}

function stackExposurePriority(className) {
  if (STACK_POSE_POLICY === "real_aspect_v1") {
    if (className === "KHR_20000") return 2;
    return 0;
  }
  if (STACK_POSE_POLICY === "real_aspect_v2") {
    if (className === "KHR_20000") return 2;
    if (className === "KHR_10000") return 1;
    if (className === "USD_50") return 1;
  }
  return 0;
}

function clampValue(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function stackPositionLimitsForClass(className) {
  if (STACK_POSE_POLICY !== "real_aspect_v2") return null;
  return STACK_REAL_ASPECT_V2_POSITION_LIMITS[className] ?? null;
}

function stackHorizontalSliceStressForClass(className, variant, index) {
  // Sparse KHR_10000 partials give geometry selection a real-aspect tail without changing default/v1 stacks.
  return STACK_POSE_POLICY === "real_aspect_v2"
    && className === "KHR_10000"
    && (variant + index) % 6 === 0;
}

function stackPositionForClass(className, position, variant, index) {
  if (stackHorizontalSliceStressForClass(className, variant, index)) {
    return [
      clampValue(position[0], -0.12, 0.12),
      (variant + index) % 12 === 0 ? 0.46 : -0.34,
      position[2],
    ];
  }
  const limits = stackPositionLimitsForClass(className);
  if (!limits) return position;
  return [
    clampValue(position[0], limits.x[0], limits.x[1]),
    clampValue(position[1], limits.y[0], limits.y[1]),
    position[2],
  ];
}

function stackLayerRanksForClasses(classNames, layerOrder) {
  const baseRanks = Array.from({ length: classNames.length }, (_, index) => layerOrder.indexOf(index));
  if (STACK_POSE_POLICY === "default") return baseRanks;
  const ordered = classNames
    .map((className, index) => ({
      index,
      priority: stackExposurePriority(className),
      baseRank: baseRanks[index],
    }))
    .sort((left, right) => (
      left.priority - right.priority
      || left.baseRank - right.baseRank
      || left.index - right.index
    ));
  const ranks = Array.from({ length: classNames.length }, () => 0);
  ordered.forEach((item, layer) => {
    ranks[item.index] = layer;
  });
  return ranks;
}

function parseCsvLine(line) {
  const values = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === "\"") {
      if (quoted && line[index + 1] === "\"") {
        value += "\"";
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      values.push(value);
      value = "";
    } else {
      value += char;
    }
  }
  values.push(value);
  return values;
}

function readAssetManifestRows() {
  const manifestPath = path.join(ASSET_BANK_DIR, "manifest.csv");
  if (!fs.existsSync(manifestPath)) return [];
  const lines = fs.readFileSync(manifestPath, "utf8").split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) return [];
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? "";
    });
    row.resolvedPath = path.resolve(ROOT, row.asset_path || "");
    return row;
  });
}

function yearRangeStart(value) {
  const match = String(value || "").match(/\d{4}/);
  return match ? Number.parseInt(match[0], 10) : 0;
}

function numberField(row, key) {
  const value = Number.parseFloat(row[key] || "0");
  return Number.isFinite(value) ? value : 0;
}

function assetQualityRank(row) {
  return [
    row.status === "in_circulation" ? 1 : 0,
    numberField(row, "max_year"),
    yearRangeStart(row.years),
    numberField(row, "width") * numberField(row, "height"),
    numberField(row, "alpha_area"),
  ];
}

function compareAssetRows(left, right) {
  const leftRank = assetQualityRank(left);
  const rightRank = assetQualityRank(right);
  for (let index = 0; index < leftRank.length; index += 1) {
    if (leftRank[index] !== rightRank[index]) return leftRank[index] - rightRank[index];
  }
  return String(left.asset_path || "").localeCompare(String(right.asset_path || ""));
}

function latestDesignRows(rows) {
  const bySide = new Map();
  for (const row of rows) {
    const side = row.side || sideForAssetPath(row.resolvedPath);
    const previous = bySide.get(side);
    if (!previous || compareAssetRows(row, previous) > 0) {
      bySide.set(side, row);
    }
  }
  return [...bySide.values()];
}

function normalizeAssetKey(assetPath) {
  return path.resolve(assetPath).toLowerCase();
}

function listFilesystemClassAssets(className) {
  const classDir = path.join(ASSET_BANK_DIR, className);
  if (!fs.existsSync(classDir)) return [];
  return fs.readdirSync(classDir)
    .filter((name) => /\.png$/i.test(name))
    .map((name) => path.join(classDir, name))
    .sort();
}

const assetManifestRows = readAssetManifestRows();
const assetManifestByPath = new Map(assetManifestRows.map((row) => [normalizeAssetKey(row.resolvedPath), row]));

function listClassAssets(className) {
  if (ASSET_QUALITY_POLICY === "filesystem_all" || assetManifestRows.length === 0) {
    return listFilesystemClassAssets(className);
  }
  const rows = assetManifestRows.filter((row) => row.class_name === className);
  const selectedRows = ASSET_QUALITY_POLICY === "latest_design" ? latestDesignRows(rows) : rows;
  return selectedRows.map((row) => row.resolvedPath).sort();
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
  const condition = noteConditionFor(asset, variant, index);
  const manifestRow = assetManifestByPath.get(normalizeAssetKey(assetPath));
  return {
    ...asset,
    path: assetPath,
    side: sideForAssetPath(assetPath),
    assetSidePolicy: ASSET_SIDE_POLICY,
    assetQualityPolicy: ASSET_QUALITY_POLICY,
    sourceStatus: manifestRow?.status || "",
    sourceYears: manifestRow?.years || "",
    sourceMaxYear: manifestRow?.max_year || "",
    sourcePath: manifestRow?.source_path || "",
    textureWidth: manifestRow?.width || "",
    textureHeight: manifestRow?.height || "",
    condition,
    printTone: notePrintToneFor(asset, variant, index, condition),
  };
}

function annotateAsset(asset, index) {
  const condition = noteConditionFor(asset, 0, index);
  const manifestRow = assetManifestByPath.get(normalizeAssetKey(asset.path));
  return {
    ...asset,
    side: sideForAssetPath(asset.path),
    assetSidePolicy: ASSET_SIDE_POLICY,
    assetQualityPolicy: ASSET_QUALITY_POLICY,
    sourceStatus: manifestRow?.status || "",
    sourceYears: manifestRow?.years || "",
    sourceMaxYear: manifestRow?.max_year || "",
    sourcePath: manifestRow?.source_path || "",
    textureWidth: manifestRow?.width || "",
    textureHeight: manifestRow?.height || "",
    condition,
    printTone: notePrintToneFor(asset, 0, index, condition),
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
  if (effectiveSceneMode === "texture_qa") return textureQaAssets(variant);
  if (effectiveSceneMode === "fan") return fanAssets(variant);
  if (effectiveSceneMode === "thin_edge") return thinEdgeAssets(variant);
  if (effectiveSceneMode === "hand_occlusion") return handOcclusionAssets(variant);
  if (effectiveSceneMode === "clean_single") return cleanSingleAssets(variant);
  if (effectiveSceneMode === "clean_context") return cleanContextAssets(variant);
  if (effectiveSceneMode === "clean") return cleanAssets(variant);
  if (variant === 0 && ASSET_SIDE_POLICY === "any" && STACK_POSE_POLICY === "default") {
    return baseAssets.map((asset, index) => annotateAsset(asset, index));
  }
  if (variant === 0 && STACK_POSE_POLICY === "default") return baseAssets.map((asset, index) => enrichAsset(asset, variant, index));
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
  const classIndexes = Array.from({ length: noteCount }, (_, index) => classIndexFor(variant, index));
  const classNames = classIndexes.map((classIndex) => CLASS_NAMES[classIndex]);
  const layerRanks = stackLayerRanksForClasses(classNames, layerOrder);
  return Array.from({ length: noteCount }, (_, index) => {
    const classIndex = classIndexes[index];
    const className = classNames[index];
    const base = baseAssets[index % baseAssets.length];
    const anchor = anchors[index % anchors.length];
    const layer = layerRanks[index];
    const position = stackPositionForClass(className, [
      anchor[0] + randomBetween(rng, -0.10, 0.10),
      anchor[1] + randomBetween(rng, -0.10, 0.10),
      0.03 + layer * 0.045,
    ], variant, index);
    const rotationX = base.rotation[0] + randomBetween(rng, -0.05, 0.05);
    const rotationY = base.rotation[1] + randomBetween(rng, -0.05, 0.05);
    const stackZRotation = stackZRotationForClass(className, base.rotation[2], rng);
    return enrichAsset({
      ...base,
      classIndex,
      className,
      idColor: INSTANCE_ID_COLORS[index],
      physicalWidthMm: PHYSICAL_WIDTH_MM[className],
      position,
      rotation: [
        rotationX,
        rotationY,
        stackZRotation,
      ],
      stackPose: {
        policy: STACK_POSE_POLICY,
        exposurePriority: stackExposurePriority(className),
        horizontalSliceStress: stackHorizontalSliceStressForClass(className, variant, index),
        positionLimits: stackPositionLimitsForClass(className),
        zRotation: round3(stackZRotation),
        zAbsRange: STACK_POSE_POLICY === "default" ? null : stackRealAspectZRanges()[className],
      },
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
  if (effectiveSceneMode === "texture_qa") return [];
  if (effectiveSceneMode === "clean" || effectiveSceneMode === "clean_single") return [];
  if (effectiveSceneMode === "clean_context") return cleanContextOccluders(variant);
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

function textureQaAssets(variant) {
  const classIndex = classIndexFor(variant, 0);
  const className = CLASS_NAMES[classIndex];
  const physicalConditionProbe = NOTE_CONDITION_POLICY === "handled_3d" && TEXTURE_QA_EFFECTS === "condition";
  const rng = mulberry32(88007111 + variant * 331 + classIndex * 17);
  return [
    enrichAsset({
      ...baseAssets[0],
      classIndex,
      className,
      idColor: INSTANCE_ID_COLORS[0],
      physicalWidthMm: PHYSICAL_WIDTH_MM[className],
      position: [0.0, 0.0, 0.012],
      rotation: [0.0, 0.0, 0.0],
      curl: physicalConditionProbe ? randomBetween(rng, 0.110, 0.190) : 0.0,
      ripple: physicalConditionProbe ? randomBetween(rng, 0.024, 0.052) : 0.0,
      roughness: 1.0,
      layer: 0,
      clean: true,
      cleanSingle: true,
      textureQa: true,
      castShadow: false,
    }, variant, 0),
  ];
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
        0.012 + index * 0.006,
      ],
      rotation: [
        randomBetween(rng, -0.045, 0.045),
        randomBetween(rng, -0.055, 0.055),
        cleanZRotationForClass(className, rng, [-0.20, 0.20]),
      ],
      curl: 0.014 + randomBetween(rng, -0.006, 0.010),
      ripple: 0.0,
      roughness: randomBetween(rng, 0.72, 0.88),
      layer: index,
      clean: true,
      castShadow: rng() < 0.22,
    }, variant, index);
  });
}

function cleanSingleAssets(variant) {
  const rng = mulberry32(26055291 + variant * 157);
  const classIndex = classIndexFor(variant, 0);
  const className = CLASS_NAMES[classIndex];
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
        0.012,
      ],
      rotation: [
        randomBetween(rng, -0.035, 0.055),
        randomBetween(rng, -0.055, 0.055),
        cleanZRotationForClass(className, rng),
      ],
      curl: 0.012 + randomBetween(rng, -0.004, 0.008),
      ripple: 0.0,
      roughness: randomBetween(rng, 0.76, 0.90),
      layer: 0,
      cleanSingle: true,
      castShadow: rng() < 0.12,
    }, variant, 0),
  ];
}

function cleanContextZRotationForClass(className, rng) {
  if (CLEAN_ORIENTATION_POLICY !== "default") {
    return cleanZRotationForClass(className, rng);
  }
  if (rng() < 0.24) {
    return randomSignedBetween(rng, 1.16, 1.55);
  }
  const base = cleanSingleZRangeForClass(className);
  const broader = {
    KHR_500: [-0.30, 0.30],
    KHR_1000: [-0.58, 0.58],
    KHR_10000: [-0.28, 0.28],
    KHR_20000: [-0.34, 0.34],
    KHR_50000: [-0.32, 0.32],
  };
  const range = broader[className] || base;
  return randomBetween(rng, range[0], range[1]);
}

function cleanContextAssets(variant) {
  const rng = mulberry32(26055417 + variant * 163);
  const classIndex = classIndexFor(variant, 0);
  const className = CLASS_NAMES[classIndex];
  return [
    enrichAsset({
      ...baseAssets[0],
      classIndex,
      className,
      idColor: INSTANCE_ID_COLORS[0],
      physicalWidthMm: PHYSICAL_WIDTH_MM[className],
      position: [
        randomBetween(rng, -0.12, 0.12),
        randomBetween(rng, -0.10, 0.10),
        0.016,
      ],
      rotation: [
        randomBetween(rng, -0.060, 0.075),
        randomBetween(rng, -0.075, 0.075),
        cleanContextZRotationForClass(className, rng),
      ],
      curl: 0.014 + randomBetween(rng, -0.005, 0.010),
      ripple: 0.0,
      roughness: randomBetween(rng, 0.74, 0.90),
      layer: 0,
      cleanContext: true,
      cleanSingle: true,
      castShadow: rng() < 0.16,
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

function normalizeAngle(angle) {
  let normalized = angle;
  while (normalized > Math.PI) normalized -= Math.PI * 2;
  while (normalized < -Math.PI) normalized += Math.PI * 2;
  return normalized;
}

function cleanZRotationForClass(className, rng, defaultRange = null) {
  if (CLEAN_ORIENTATION_POLICY === "default") {
    const range = defaultRange || cleanSingleZRangeForClass(className);
    return randomBetween(rng, range[0], range[1]);
  }
  const ranges = CLEAN_ORIENTATION_POLICY === "real_aspect_bridge_v1"
    ? CLEAN_REAL_ASPECT_BRIDGE_Z_ABS_RANGES
    : CLEAN_ORIENTATION_POLICY === "real_aspect_square_v1"
      ? CLEAN_REAL_ASPECT_SQUARE_Z_ABS_RANGES
      : STACK_REAL_ASPECT_Z_ABS_RANGES;
  const range = ranges[className];
  if (!range) {
    throw new Error(`No clean real-aspect z-rotation range for class ${className}`);
  }
  let angle = randomSignedBetween(rng, range[0], range[1]);
  if (rng() < 0.12) angle += Math.PI;
  return normalizeAngle(angle);
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

const NEGATIVE_PROP_STYLES = [
  { propKind: "receipt", textureStyle: "receipt", colors: [0xf4efe3, 0xe7dfcf], width: [0.48, 0.92], height: [0.14, 0.30] },
  { propKind: "payment_card", textureStyle: "payment_card", colors: [0x2f5570, 0xd7e3ec, 0x5f6f61], width: [0.36, 0.62], height: [0.20, 0.34] },
  { propKind: "wallet", textureStyle: "wallet", colors: [0x2c2520, 0x5a3d2e, 0x283241], width: [0.48, 0.86], height: [0.24, 0.42] },
  { propKind: "blank_paper", textureStyle: "paper", colors: [0xf1ead8, 0xd8d2c4, 0xd7e3ec], width: [0.42, 0.84], height: [0.18, 0.36] },
  { propKind: "phone", textureStyle: "phone", colors: [0x17202b, 0x263447, 0x0f141c], width: [0.26, 0.42], height: [0.54, 0.86] },
  { propKind: "sticky_note", textureStyle: "sticky_note", colors: [0xe6d1a3, 0xb6c59a, 0xf2df8f], width: [0.20, 0.42], height: [0.18, 0.34] },
];

const UNKNOWN_CURRENCY_PROP_STYLES = [
  { propKind: "unknown_banknote", textureStyle: "unknown_banknote", negativeConfusionHardness: "hard", colors: [0xd8b5a7, 0xc8d3b7, 0xb8c7d9, 0xd6c2da], width: [0.60, 0.98], height: [0.24, 0.42] },
  { propKind: "unknown_banknote", textureStyle: "unknown_banknote", negativeConfusionHardness: "hard", colors: [0xe4c0a2, 0xb7d2ca, 0xd3c5a6, 0xc6b7d5], width: [0.54, 0.88], height: [0.22, 0.38] },
  { propKind: "coin_cluster", textureStyle: "coin_cluster", negativeConfusionHardness: "coin", colors: [0x76634f, 0x88775d, 0x605447], width: [0.34, 0.66], height: [0.28, 0.58] },
  { propKind: "receipt", textureStyle: "receipt", negativeConfusionHardness: "none", colors: [0xf4efe3, 0xe7dfcf], width: [0.48, 0.82], height: [0.14, 0.28] },
  { propKind: "blank_paper", textureStyle: "paper", negativeConfusionHardness: "none", colors: [0xf1ead8, 0xd8d2c4, 0xd7e3ec], width: [0.42, 0.76], height: [0.18, 0.34] },
  { propKind: "payment_card", textureStyle: "payment_card", negativeConfusionHardness: "none", colors: [0x2f5570, 0xd7e3ec, 0x5f6f61], width: [0.34, 0.56], height: [0.19, 0.32] },
];

const UNKNOWN_CURRENCY_SOFT_PROP_STYLES = [
  { propKind: "unknown_banknote", textureStyle: "unknown_banknote", negativeConfusionHardness: "soft", colors: [0xd9d0b5, 0xc9d8d0, 0xd4c9d6, 0xcdd5df], width: [0.46, 0.74], height: [0.18, 0.30] },
  { propKind: "unknown_banknote", textureStyle: "unknown_banknote", negativeConfusionHardness: "soft", colors: [0xe2d7bf, 0xbfd2c2, 0xd7cabb, 0xc6d2d4], width: [0.42, 0.68], height: [0.17, 0.28] },
  { propKind: "coin_cluster", textureStyle: "coin_cluster", negativeConfusionHardness: "coin", colors: [0x76634f, 0x88775d, 0x605447], width: [0.30, 0.56], height: [0.24, 0.50] },
  { propKind: "receipt", textureStyle: "receipt", negativeConfusionHardness: "none", colors: [0xf4efe3, 0xe7dfcf], width: [0.48, 0.82], height: [0.14, 0.28] },
  { propKind: "blank_paper", textureStyle: "paper", negativeConfusionHardness: "none", colors: [0xf1ead8, 0xd8d2c4, 0xd7e3ec], width: [0.42, 0.76], height: [0.18, 0.34] },
  { propKind: "payment_card", textureStyle: "payment_card", negativeConfusionHardness: "none", colors: [0x2f5570, 0xd7e3ec, 0x5f6f61], width: [0.34, 0.56], height: [0.19, 0.32] },
];

const UNKNOWN_CURRENCY_SOFT_DARK_PROP_STYLES = [
  { propKind: "unknown_banknote", textureStyle: "unknown_banknote", negativeConfusionHardness: "soft", negativeVisualDomain: "mined_fp_dark_v1", colors: [0x321826, 0x20392d, 0x1e2d4a, 0x482818], width: [0.46, 0.74], height: [0.18, 0.30] },
  { propKind: "unknown_banknote", textureStyle: "unknown_banknote", negativeConfusionHardness: "soft", negativeVisualDomain: "mined_fp_dark_v1", colors: [0x4c202e, 0x174048, 0x28381f, 0x3c244c], width: [0.42, 0.68], height: [0.17, 0.28] },
  { propKind: "coin_cluster", textureStyle: "coin_cluster", negativeConfusionHardness: "coin", colors: [0x453427, 0x55432e, 0x2e2922], width: [0.30, 0.56], height: [0.24, 0.50] },
  { propKind: "receipt", textureStyle: "receipt", negativeConfusionHardness: "none", colors: [0x625748, 0x4a4237], width: [0.48, 0.82], height: [0.14, 0.28] },
  { propKind: "blank_paper", textureStyle: "paper", negativeConfusionHardness: "none", colors: [0x575248, 0x3d3a34, 0x42494d], width: [0.42, 0.76], height: [0.18, 0.34] },
  { propKind: "payment_card", textureStyle: "payment_card", negativeConfusionHardness: "none", colors: [0x142434, 0x263427, 0x351f31], width: [0.34, 0.56], height: [0.19, 0.32] },
];

const UNKNOWN_CURRENCY_SPREAD_DARK_PROP_STYLES = [
  { propKind: "unknown_banknote", textureStyle: "unknown_banknote", negativeConfusionHardness: "partial", negativeVisualDomain: "mined_fp_dark_v1", colors: [0x342018, 0x1c3440, 0x2b3c1d, 0x3e244b], width: [0.58, 0.92], height: [0.22, 0.38] },
  { propKind: "receipt", textureStyle: "receipt", negativeConfusionHardness: "none", colors: [0x655a4a, 0x514638, 0x463f36], width: [0.48, 0.86], height: [0.14, 0.30] },
  { propKind: "payment_card", textureStyle: "payment_card", negativeConfusionHardness: "none", colors: [0x142434, 0x263427, 0x351f31], width: [0.34, 0.58], height: [0.19, 0.34] },
  { propKind: "patterned_paper", textureStyle: "paper", negativeConfusionHardness: "none", colors: [0x4b4740, 0x353a3d, 0x463a42], width: [0.42, 0.78], height: [0.18, 0.36] },
  { propKind: "coin_cluster", textureStyle: "coin_cluster", negativeConfusionHardness: "coin", colors: [0x453427, 0x55432e, 0x2e2922], width: [0.30, 0.56], height: [0.24, 0.50] },
  { propKind: "unknown_banknote", textureStyle: "unknown_banknote", negativeConfusionHardness: "soft", negativeVisualDomain: "mined_fp_dark_v1", colors: [0x482818, 0x174048, 0x28381f, 0x3c244c], width: [0.36, 0.64], height: [0.15, 0.27] },
  { propKind: "wallet", textureStyle: "wallet", negativeConfusionHardness: "none", colors: [0x1f1714, 0x33251c, 0x171f28], width: [0.42, 0.74], height: [0.22, 0.40] },
];

const UNKNOWN_CURRENCY_FULLFRAME_PROP_STYLES = [
  { propKind: "unknown_banknote", textureStyle: "unknown_banknote", negativeConfusionHardness: "fullframe", colors: [0xd1b7a6, 0xb9cbbb, 0xb8c8dd, 0xd7c0cf], width: [1.02, 1.34], height: [0.40, 0.62] },
  { propKind: "unknown_banknote", textureStyle: "unknown_banknote", negativeConfusionHardness: "fullframe", colors: [0xe0c1a0, 0xb8d2c8, 0xd5c7a6, 0xc8bad8], width: [0.92, 1.22], height: [0.36, 0.56] },
  { propKind: "unknown_banknote", textureStyle: "unknown_banknote", negativeConfusionHardness: "fullframe", colors: [0xc7d3bc, 0xd8cab0, 0xb9c9d8, 0xd5bdd0], width: [0.84, 1.12], height: [0.34, 0.52] },
];

const UNKNOWN_CURRENCY_FULLFRAME_DARK_PROP_STYLES = [
  { propKind: "unknown_banknote", textureStyle: "unknown_banknote", negativeConfusionHardness: "fullframe", negativeVisualDomain: "mined_fp_dark_v1", colors: [0x321826, 0x20392d, 0x1e2d4a, 0x482818], width: [1.02, 1.36], height: [0.40, 0.64] },
  { propKind: "unknown_banknote", textureStyle: "unknown_banknote", negativeConfusionHardness: "fullframe", negativeVisualDomain: "mined_fp_dark_v1", colors: [0x4c202e, 0x174048, 0x28381f, 0x3c244c], width: [0.92, 1.24], height: [0.36, 0.58] },
  { propKind: "unknown_banknote", textureStyle: "unknown_banknote", negativeConfusionHardness: "fullframe", negativeVisualDomain: "mined_fp_dark_v1", colors: [0x241a12, 0x1a2735, 0x30213d, 0x17362b], width: [0.86, 1.16], height: [0.34, 0.54] },
];

function isFullFrameNegativePolicy() {
  return NEGATIVE_PROP_POLICY === "unknown_currency_fullframe_v1" || NEGATIVE_PROP_POLICY === "unknown_currency_fullframe_dark_v1";
}

function isSpreadDarkNegativePolicy() {
  return NEGATIVE_PROP_POLICY === "unknown_currency_spread_dark_v1";
}

function negativePropStylesForPolicy() {
  if (isSpreadDarkNegativePolicy()) return UNKNOWN_CURRENCY_SPREAD_DARK_PROP_STYLES;
  if (NEGATIVE_PROP_POLICY === "unknown_currency_soft_dark_v1") return UNKNOWN_CURRENCY_SOFT_DARK_PROP_STYLES;
  if (NEGATIVE_PROP_POLICY === "unknown_currency_soft_v1") return UNKNOWN_CURRENCY_SOFT_PROP_STYLES;
  if (isFullFrameNegativePolicy()) return UNKNOWN_CURRENCY_SOFT_PROP_STYLES;
  return NEGATIVE_PROP_POLICY === "unknown_currency_v1" ? UNKNOWN_CURRENCY_PROP_STYLES : NEGATIVE_PROP_STYLES;
}

function isSoftUnknownCurrencyPolicy() {
  return NEGATIVE_PROP_POLICY === "unknown_currency_soft_v1" || NEGATIVE_PROP_POLICY === "unknown_currency_soft_dark_v1";
}

const CLEAN_CONTEXT_PROP_PLACEMENTS = [
  { styleIndex: 4, position: [-0.76, 0.02], zRotation: 0.06, maxWidth: 0.28, maxHeight: 0.58 },
  { styleIndex: 0, position: [0.06, 0.60], zRotation: -0.08, maxWidth: 0.64, maxHeight: 0.16 },
  { styleIndex: 1, position: [0.76, -0.28], zRotation: 0.30, maxWidth: 0.36, maxHeight: 0.22 },
  { styleIndex: 2, position: [-0.72, -0.46], zRotation: -0.24, maxWidth: 0.44, maxHeight: 0.22 },
  { styleIndex: 5, position: [0.74, 0.40], zRotation: -0.20, maxWidth: 0.26, maxHeight: 0.20 },
  { styleIndex: 3, position: [-0.10, -0.60], zRotation: 0.10, maxWidth: 0.56, maxHeight: 0.16 },
];

function cleanContextOccluders(variant) {
  const rng = mulberry32(26055519 + variant * 167);
  const propCount = 2 + (variant % 3);
  const props = [];
  const start = variant % CLEAN_CONTEXT_PROP_PLACEMENTS.length;
  for (let index = 0; index < propCount; index += 1) {
    const placement = CLEAN_CONTEXT_PROP_PLACEMENTS[(start + index) % CLEAN_CONTEXT_PROP_PLACEMENTS.length];
    const style = NEGATIVE_PROP_STYLES[placement.styleIndex];
    props.push({
      kind: "cover_card",
      propKind: style.propKind,
      textureStyle: style.textureStyle,
      seed: 26055519 + variant * 167 + index * 977,
      layer: 12 + index,
      color: style.colors[randomInt(rng, style.colors.length)],
      position: [
        placement.position[0] + randomBetween(rng, -0.035, 0.035),
        placement.position[1] + randomBetween(rng, -0.030, 0.030),
        0.026 + index * 0.006,
      ],
      rotation: [
        randomBetween(rng, -0.05, 0.06),
        randomBetween(rng, -0.05, 0.06),
        placement.zRotation + randomBetween(rng, -0.20, 0.20),
      ],
      width: Math.min(randomBetween(rng, style.width[0], style.width[1]), placement.maxWidth),
      height: Math.min(randomBetween(rng, style.height[0], style.height[1]), placement.maxHeight),
      contextOnly: true,
    });
  }
  if (variant % 5 === 0) {
    props.push({
      kind: "finger_capsule",
      layer: 18,
      color: [0xc58663, 0xb87958, 0xd09a77][randomInt(rng, 3)],
      position: [
        randomBetween(rng, -0.80, -0.68),
        randomBetween(rng, -0.58, -0.44),
        0.044,
      ],
      rotation: [
        randomBetween(rng, 0.02, 0.16),
        randomBetween(rng, -0.10, 0.10),
        randomBetween(rng, 0.35, 1.10),
      ],
      radius: randomBetween(rng, 0.034, 0.048),
      length: randomBetween(rng, 0.28, 0.48),
      contextOnly: true,
    });
  }
  return props;
}

function negativeOccluders(variant) {
  const rng = mulberry32(26057511 + variant * 149);
  const propCount = isFullFrameNegativePolicy() ? 2 + (variant % 3) : isSpreadDarkNegativePolicy() ? 4 + (variant % 3) : 3 + (variant % 3);
  const styles = negativePropStylesForPolicy();
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
    const isFullFrameUnknown = isFullFrameNegativePolicy() && index === 0;
    const softUnknownStyles = NEGATIVE_PROP_POLICY === "unknown_currency_soft_dark_v1"
      ? UNKNOWN_CURRENCY_SOFT_DARK_PROP_STYLES
      : UNKNOWN_CURRENCY_SOFT_PROP_STYLES;
    const style = isFullFrameUnknown
      ? NEGATIVE_PROP_POLICY === "unknown_currency_fullframe_dark_v1"
        ? UNKNOWN_CURRENCY_FULLFRAME_DARK_PROP_STYLES[variant % UNKNOWN_CURRENCY_FULLFRAME_DARK_PROP_STYLES.length]
        : UNKNOWN_CURRENCY_FULLFRAME_PROP_STYLES[variant % UNKNOWN_CURRENCY_FULLFRAME_PROP_STYLES.length]
      : NEGATIVE_PROP_POLICY === "unknown_currency_v1" && index === 0
      ? UNKNOWN_CURRENCY_PROP_STYLES[variant % 2]
      : isSoftUnknownCurrencyPolicy() && index === 0 && variant % 2 === 0
        ? softUnknownStyles[variant % 2]
      : styles[(variant + index * 3) % styles.length];
    const spreadUnknown = isSpreadDarkNegativePolicy() && style.propKind === "unknown_banknote";
    const partialOffcenter = spreadUnknown && style.negativeConfusionHardness === "partial";
    const offcenterSign = variant % 2 === 0 ? -1 : 1;
    props.push({
      kind: "cover_card",
      propKind: style.propKind,
      textureStyle: style.textureStyle,
      negativeConfusionHardness: style.negativeConfusionHardness || "none",
      negativeVisualDomain: style.negativeVisualDomain || "default",
      seed: 26057511 + variant * 149 + index * 811,
      layer: 20 + index,
      color: style.colors[randomInt(rng, style.colors.length)],
      position: [
        isFullFrameUnknown
          ? randomBetween(rng, -0.14, 0.14)
          : partialOffcenter
            ? offcenterSign * randomBetween(rng, 0.34, 0.74)
            : spreadUnknown
              ? randomBetween(rng, -0.76, 0.76)
              : randomBetween(rng, -0.66, 0.66),
        isFullFrameUnknown
          ? randomBetween(rng, -0.12, 0.12)
          : partialOffcenter
            ? randomBetween(rng, -0.50, 0.50)
            : spreadUnknown
              ? randomBetween(rng, -0.52, 0.52)
              : randomBetween(rng, -0.44, 0.44),
        0.05 + index * 0.01,
      ],
      rotation: [
        isFullFrameUnknown ? randomBetween(rng, -0.05, 0.07) : randomBetween(rng, -0.10, 0.10),
        isFullFrameUnknown ? randomBetween(rng, -0.06, 0.06) : randomBetween(rng, -0.10, 0.10),
        isFullFrameUnknown ? randomBetween(rng, -1.30, 1.30) : spreadUnknown ? randomBetween(rng, -1.75, 1.75) : randomBetween(rng, -1.55, 1.55),
      ],
      width: randomBetween(rng, style.width[0], style.width[1]),
      height: randomBetween(rng, style.height[0], style.height[1]),
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

function applyOccluderPolicy(occluders) {
  if (OCCLUDER_POLICY === "scene_default") return occluders;
  if (OCCLUDER_POLICY === "none") return [];
  return occluders.filter((occluder) => {
    const kind = String(occluder.kind ?? "").toLowerCase();
    return !kind.includes("finger") && !kind.includes("hand");
  });
}

const backgroundFiles = listImageFiles(BACKGROUND_DIR);
const environmentFiles = listEnvironmentFiles(ENVIRONMENT_DIR);

function buildRenderState(variant, outDir) {
  effectiveSceneMode = effectiveSceneModeFor(variant, SCENE_MODE);
  const assets = variantAssets(variant);
  const sceneDefaultOccluders = variantOccluders(variant);
  const occluders = applyOccluderPolicy(sceneDefaultOccluders);
  const selectedBackgroundPath = backgroundFiles.length ? backgroundFiles[variant % backgroundFiles.length] : null;
  const selectedEnvironmentPath = environmentFiles.length ? environmentFiles[variant % environmentFiles.length] : null;
  const config = sceneConfig(variant, effectiveSceneMode, selectedBackgroundPath, selectedEnvironmentPath);
  const assetSideCounts = assets.reduce((counts, asset) => {
    const side = asset.side ?? "unknown";
    counts[side] = (counts[side] ?? 0) + 1;
    return counts;
  }, {});
  const frontBackMixSatisfied = assetSideCounts.front > 0 && assetSideCounts.back > 0;
  const textureAssets = assets.map((asset) => ({ ...asset, textureUrl: pathToFileURL(asset.path).href }));
  return {
    outDir,
    variant,
    effectiveSceneMode,
    assets,
    textureAssets,
    sceneDefaultOccluders,
    occluders,
    config,
    assetSideCounts,
    frontBackMixSatisfied,
  };
}

function html(renderState) {
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

const assets = ${JSON.stringify(renderState.textureAssets)};
const occluders = ${JSON.stringify(renderState.occluders)};
const sceneConfig = ${JSON.stringify(renderState.config)};
const WIDTH = ${JSON.stringify(WIDTH)};
const HEIGHT = ${JSON.stringify(HEIGHT)};
const VISUAL_SCALE = ${JSON.stringify(VISUAL_SCALE)};
const MIN_VISIBLE_PIXELS = ${JSON.stringify(MIN_VISIBLE_PIXELS)};
const NEGATIVE_PROP_LABEL_POLICY = ${JSON.stringify(NEGATIVE_PROP_LABEL_POLICY)};
const UNKNOWN_PROP_CLASS_NAME = ${JSON.stringify(UNKNOWN_PROP_CLASS_NAME)};
const UNKNOWN_PROP_CLASS_INDEX = ${JSON.stringify(UNKNOWN_PROP_CLASS_INDEX)};
const UNKNOWN_PROP_ID_CHANNELS = ${JSON.stringify(UNKNOWN_PROP_ID_CHANNELS)};
const scene = new THREE.Scene();
scene.background = new THREE.Color(sceneConfig.surface.scene);

const camera = new THREE.PerspectiveCamera(sceneConfig.camera.fov, WIDTH / HEIGHT, 0.01, 20);
camera.position.set(...sceneConfig.camera.position);
camera.lookAt(...sceneConfig.camera.lookAt);

const visualRenderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
visualRenderer.setPixelRatio(VISUAL_SCALE);
visualRenderer.setSize(WIDTH, HEIGHT);
visualRenderer.shadowMap.enabled = true;
visualRenderer.shadowMap.type = THREE.VSMShadowMap;
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

function pixelNoise01(x, y, seed) {
  let hash = (Math.imul((x + 0x9e3779b9) >>> 0, 0x85ebca6b) ^ Math.imul((y + 0xc2b2ae35) >>> 0, 0x27d4eb2d) ^ (seed >>> 0)) >>> 0;
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x7feb352d) >>> 0;
  hash ^= hash >>> 15;
  hash = Math.imul(hash, 0x846ca68b) >>> 0;
  hash ^= hash >>> 16;
  return (hash >>> 0) / 0x100000000;
}

function buildCameraOverlay() {
  const context = grainCanvas.getContext("2d", { willReadFrequently: true });
  const image = context.createImageData(grainCanvas.width, grainCanvas.height);
  const cx = grainCanvas.width / 2;
  const cy = grainCanvas.height / 2;
  const maxRadius = Math.sqrt(cx * cx + cy * cy);
  const grainSeed = sceneConfig.postprocess.grainSeed || 1;
  for (let y = 0; y < grainCanvas.height; y += 1) {
    for (let x = 0; x < grainCanvas.width; x += 1) {
      const i = (y * grainCanvas.width + x) * 4;
      const grain = (pixelNoise01(x, y, grainSeed) - 0.5) * sceneConfig.postprocess.grainStrength;
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
key.shadow.mapSize.set(4096, 4096);
key.shadow.camera.left = -2.2;
key.shadow.camera.right = 2.2;
key.shadow.camera.top = 1.6;
key.shadow.camera.bottom = -1.6;
key.shadow.camera.near = 0.2;
key.shadow.camera.far = 8.0;
key.shadow.radius = 4;
key.shadow.blurSamples = 8;
key.shadow.bias = -0.00008;
key.shadow.normalBias = 0.018;
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
  sceneConfig.textureQa
    ? new THREE.MeshBasicMaterial({ color: sceneConfig.surface.scene, map: makeTableTexture() })
    : new THREE.MeshStandardMaterial({ color: 0xffffff, map: makeTableTexture(), roughness: 0.88 })
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
  if (sceneConfig.surface.flatTexture) {
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(...sceneConfig.surface.repeat);
    texture.needsUpdate = true;
    return texture;
  }
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
  const creaseCount = Math.max(0, Math.min(4, Math.round(condition.creaseCount ?? 0)));
  const creaseStrength = Math.min(1.0, crinkle * 0.75 + creaseCount * 0.08);
  const curlScale = Math.max(0.0, condition.geometryCurlScale ?? 1.0);
  const rippleScale = Math.max(0.0, condition.geometryRippleScale ?? 1.0);
  const wrinkleScale = Math.max(0.0, condition.geometryWrinkleScale ?? 1.0);
  const creaseScale = Math.max(0.0, condition.geometryCreaseScale ?? 1.0);
  const creaseWidth = Math.max(0.0008, Math.min(0.008, condition.geometryCreaseWidth ?? 0.0028));
  const shoulderDistance = Math.max(0.030, Math.min(0.090, Math.sqrt(creaseWidth) * 1.55));
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const base = curl * curlScale * x * x + ripple * rippleScale * Math.sin(x * 10.0) * Math.sin((y + 0.34) * 7.0);
    const wrinkles = crinkle * 0.0065 * wrinkleScale * (
      Math.sin(x * 28.0 + phase) * Math.sin(y * 17.0 - phase * 0.7)
      + 0.5 * Math.sin((x + y) * 36.0 + phase * 1.7)
    );
    let crease = 0.0;
    for (let creaseIndex = 0; creaseIndex < creaseCount; creaseIndex += 1) {
      const centeredIndex = creaseIndex - (creaseCount - 1) / 2;
      const localAngle = creaseAngle + centeredIndex * 0.34 + Math.sin(phase + creaseIndex * 1.17) * 0.08;
      const localOffset = creaseOffset + centeredIndex * 0.16 + Math.sin(phase * 0.7 + creaseIndex * 1.31) * 0.035;
      const creaseCos = Math.cos(localAngle);
      const creaseSin = Math.sin(localAngle);
      const creaseDistance = x * creaseCos + y * creaseSin - localOffset;
      const core = Math.exp(-(creaseDistance * creaseDistance) / creaseWidth);
      const shoulder = Math.exp(-Math.pow(Math.abs(creaseDistance) - shoulderDistance, 2) / (creaseWidth * 2.4));
      const polarity = creaseIndex % 2 === 0 ? 1.0 : -0.82;
      const localStrength = creaseStrength * (1.0 - Math.min(0.28, Math.abs(centeredIndex) * 0.10));
      crease += polarity * localStrength * 0.012 * creaseScale * (core - shoulder * 0.38);
    }
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

function numberToCssHex(value) {
  return "#" + Math.max(0, Number(value || 0)).toString(16).padStart(6, "0").slice(-6);
}

function makeOccluderTexture(occluder) {
  if (!occluder.textureStyle) return null;
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const context = canvas.getContext("2d");
  const rng = browserMulberry32(occluder.seed || 1);
  context.fillStyle = numberToCssHex(occluder.color);
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = "rgba(28,24,20,0.25)";
  context.lineWidth = 10;
  context.strokeRect(5, 5, canvas.width - 10, canvas.height - 10);

  if (occluder.textureStyle === "receipt") {
    context.fillStyle = "rgba(255,255,255,0.38)";
    context.fillRect(24, 22, 464, 468);
    context.fillStyle = "rgba(38,38,38,0.42)";
    for (let y = 72; y < 430; y += 32 + rng() * 10) {
      const x = 38 + rng() * 28;
      context.fillRect(x, y, 170 + rng() * 230, 7 + rng() * 5);
    }
    context.fillStyle = "rgba(38,38,38,0.20)";
    context.fillRect(56, 438, 260, 30);
  } else if (occluder.textureStyle === "payment_card") {
    context.fillStyle = "rgba(10,16,22,0.62)";
    context.fillRect(0, 86, 512, 70);
    context.fillStyle = "rgba(232,206,134,0.72)";
    context.fillRect(62, 224, 96, 70);
    context.fillStyle = "rgba(255,255,255,0.34)";
    context.fillRect(56, 350, 210, 14);
    context.fillRect(56, 386, 152, 12);
  } else if (occluder.textureStyle === "unknown_banknote") {
    const fullframe = occluder.negativeConfusionHardness === "fullframe";
    const soft = occluder.negativeConfusionHardness === "soft";
    const darkTeacher = occluder.negativeVisualDomain === "mined_fp_dark_v1";
    const accent = darkTeacher
      ? ["rgba(184,24,76,0.62)", "rgba(12,132,170,0.58)", "rgba(84,168,28,0.54)", "rgba(132,54,196,0.54)"][Math.floor(rng() * 4)]
      : ["rgba(122,52,78,0.34)", "rgba(42,96,116,0.32)", "rgba(74,118,66,0.30)"][Math.floor(rng() * 3)];
    context.fillStyle = soft
      ? "rgba(255,255,255,0.18)"
      : darkTeacher
        ? "rgba(255,255,255,0.08)"
        : fullframe
          ? "rgba(255,255,255,0.20)"
          : "rgba(255,255,255,0.24)";
    context.fillRect(22, 24, 468, 464);
    context.strokeStyle = soft
      ? "rgba(72,64,54,0.18)"
      : darkTeacher
        ? "rgba(4,3,3,0.36)"
        : fullframe
          ? "rgba(72,54,48,0.24)"
          : "rgba(72,54,48,0.28)";
    context.lineWidth = soft ? 6 : fullframe ? 9 : 8;
    context.strokeRect(34, 36, 444, 440);
    context.strokeStyle = accent;
    context.lineWidth = soft ? 3 : darkTeacher ? 5 : fullframe ? 4 : 5;
    for (let y = 76; y < 450; y += soft ? 68 : fullframe ? 34 : 42) {
      context.beginPath();
      for (let x = 54; x <= 458; x += soft ? 28 : fullframe ? 16 : 18) {
        const wave = Math.sin((x + y) * (soft ? 0.022 : fullframe ? 0.040 : 0.035) + rng() * 0.4) * (soft ? 5 : darkTeacher ? 10 : fullframe ? 7 : 9);
        if (x === 54) context.moveTo(x, y + wave);
        else context.lineTo(x, y + wave);
      }
      context.stroke();
    }
    if (fullframe) {
      context.save();
      context.translate(248, 256);
      context.rotate(-0.08);
      context.fillStyle = darkTeacher ? "rgba(255,255,255,0.07)" : "rgba(255,255,255,0.18)";
      context.fillRect(-28, -218, 56, 436);
      context.strokeStyle = darkTeacher ? "rgba(0,0,0,0.30)" : "rgba(68,58,48,0.24)";
      context.lineWidth = 4;
      context.strokeRect(-28, -218, 56, 436);
      context.restore();
      context.fillStyle = darkTeacher ? "rgba(220,230,240,0.12)" : "rgba(35,32,30,0.20)";
      for (let y = 112; y < 396; y += 44) {
        context.fillRect(292 + rng() * 10, y, 118 + rng() * 54, 6);
        context.fillRect(292 + rng() * 10, y + 16, 76 + rng() * 38, 5);
      }
      if (darkTeacher) {
        context.fillStyle = "rgba(190,28,74,0.34)";
        context.fillRect(44, 52, 146, 388);
        context.fillStyle = "rgba(10,128,184,0.31)";
        context.fillRect(190, 70, 118, 360);
        context.fillStyle = "rgba(72,160,28,0.29)";
        context.fillRect(318, 52, 132, 386);
      }
    }
    context.fillStyle = darkTeacher ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.28)";
    context.beginPath();
    context.ellipse(soft ? 346 : 334, 242, soft ? 72 : 92, soft ? 104 : 132, -0.12, 0, Math.PI * 2);
    context.fill();
    context.strokeStyle = soft ? "rgba(68,58,48,0.18)" : darkTeacher ? "rgba(0,0,0,0.38)" : "rgba(68,50,48,0.28)";
    context.lineWidth = soft ? 4 : 6;
    context.stroke();
    context.fillStyle = accent;
    context.beginPath();
    context.ellipse(122, 128, soft ? 52 : 62, soft ? 30 : 40, 0.22, 0, Math.PI * 2);
    context.fill();
    context.beginPath();
    context.arc(122, 128, soft ? 18 : 26, 0, Math.PI * 2);
    context.stroke();
    if (soft) {
      context.save();
      context.translate(252, 260);
      context.rotate(-0.55);
      context.fillStyle = "rgba(255,255,255,0.20)";
      context.fillRect(-230, -34, 460, 68);
      context.restore();
    }
    context.fillStyle = soft ? "rgba(55,48,42,0.36)" : darkTeacher ? "rgba(235,225,210,0.30)" : "rgba(55,44,40,0.50)";
    context.font = soft ? "bold 56px sans-serif" : fullframe ? "bold 86px sans-serif" : "bold 78px sans-serif";
    const denom = (soft ? ["40", "70", "90", "250", "880"] : fullframe ? ["30", "70", "300", "700", "900"] : ["25", "75", "200", "300", "700"])[Math.floor(rng() * 5)];
    context.fillText(denom, 54, 372);
    context.font = soft ? "bold 26px sans-serif" : fullframe ? "bold 36px sans-serif" : "bold 34px sans-serif";
    context.fillText(["RT", "BN", "LM", "QA"][Math.floor(rng() * 4)], 360, 92);
    context.fillStyle = darkTeacher ? "rgba(235,225,210,0.20)" : "rgba(35,32,30,0.24)";
    for (let i = 0; i < (soft ? 4 : fullframe ? 9 : 7); i += 1) {
      context.fillRect(78 + i * (soft ? 52 : fullframe ? 30 : 34), 420 + (rng() - 0.5) * 8, 18 + rng() * (soft ? 24 : 34), soft ? 5 : fullframe ? 6 : 7);
    }
    if (darkTeacher) {
      context.fillStyle = "rgba(0,0,0,0.12)";
      context.fillRect(0, 0, 512, 512);
    }
  } else if (occluder.textureStyle === "coin_cluster") {
    context.fillStyle = "rgba(65,54,42,0.34)";
    context.fillRect(20, 18, 472, 474);
    for (let i = 0; i < 18; i += 1) {
      const x = 74 + rng() * 360;
      const y = 68 + rng() * 370;
      const radius = 24 + rng() * 35;
      const metal = [
        "rgba(202,184,132,0.86)",
        "rgba(176,119,68,0.86)",
        "rgba(196,199,190,0.88)",
      ][Math.floor(rng() * 3)];
      context.fillStyle = metal;
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();
      context.strokeStyle = "rgba(44,38,32,0.38)";
      context.lineWidth = 5;
      context.stroke();
      context.strokeStyle = "rgba(255,255,255,0.22)";
      context.lineWidth = 3;
      context.beginPath();
      context.arc(x, y, radius * 0.62, 0, Math.PI * 2);
      context.stroke();
    }
  } else if (occluder.textureStyle === "wallet") {
    context.fillStyle = "rgba(0,0,0,0.20)";
    context.fillRect(28, 46, 456, 420);
    context.strokeStyle = "rgba(230,210,170,0.24)";
    context.lineWidth = 12;
    context.strokeRect(52, 76, 408, 360);
    context.beginPath();
    context.moveTo(92, 96);
    context.lineTo(430, 420);
    context.stroke();
  } else if (occluder.textureStyle === "phone") {
    context.fillStyle = "rgba(0,0,0,0.38)";
    context.fillRect(32, 28, 448, 456);
    context.strokeStyle = "rgba(200,220,238,0.24)";
    context.lineWidth = 8;
    context.strokeRect(52, 54, 408, 408);
    context.fillStyle = "rgba(230,245,255,0.10)";
    context.fillRect(110, 92, 150, 34);
  } else if (occluder.textureStyle === "sticky_note") {
    context.fillStyle = "rgba(255,255,255,0.18)";
    context.fillRect(22, 22, 468, 438);
    context.fillStyle = "rgba(80,70,32,0.16)";
    context.beginPath();
    context.moveTo(376, 460);
    context.lineTo(490, 460);
    context.lineTo(490, 350);
    context.closePath();
    context.fill();
  } else if (occluder.textureStyle === "paper") {
    context.fillStyle = "rgba(255,255,255,0.26)";
    context.fillRect(20, 18, 472, 474);
    context.strokeStyle = "rgba(70,70,70,0.16)";
    context.lineWidth = 7;
    for (let y = 92; y < 410; y += 58) {
      context.beginPath();
      context.moveTo(60, y + (rng() - 0.5) * 10);
      context.lineTo(442, y + (rng() - 0.5) * 10);
      context.stroke();
    }
    context.fillStyle = "rgba(90,80,60,0.15)";
    context.beginPath();
    context.moveTo(380, 492);
    context.lineTo(492, 492);
    context.lineTo(492, 380);
    context.closePath();
    context.fill();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
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

function clampByte(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function applyPrintToneAdjustment(canvas, printTone) {
  if (!printTone || printTone.policy === "off") return;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const image = context.getImageData(0, 0, canvas.width, canvas.height);
  const data = image.data;
  const contrast = Number.isFinite(printTone.contrast) ? printTone.contrast : 1.0;
  const saturation = Number.isFinite(printTone.saturation) ? printTone.saturation : 1.0;
  const brightness = Number.isFinite(printTone.brightness) ? printTone.brightness : 0.0;
  const shadowPull = Number.isFinite(printTone.shadowPull) ? printTone.shadowPull : 0.0;
  const highlightPush = Number.isFinite(printTone.highlightPush) ? printTone.highlightPush : 0.0;
  for (let offset = 0; offset < data.length; offset += 4) {
    const alpha = data[offset + 3] / 255;
    if (alpha <= 0) continue;
    let red = data[offset];
    let green = data[offset + 1];
    let blue = data[offset + 2];
    const luma = red * 0.2126 + green * 0.7152 + blue * 0.0722;
    red = luma + (red - luma) * saturation;
    green = luma + (green - luma) * saturation;
    blue = luma + (blue - luma) * saturation;
    const adjustedLuma = luma < 128
      ? -shadowPull * (1 - luma / 128)
      : highlightPush * ((luma - 128) / 127);
    data[offset] = clampByte((red - 128) * contrast + 128 + brightness + adjustedLuma);
    data[offset + 1] = clampByte((green - 128) * contrast + 128 + brightness + adjustedLuma);
    data[offset + 2] = clampByte((blue - 128) * contrast + 128 + brightness + adjustedLuma);
  }
  context.putImageData(image, 0, 0);
}

function drawConditionedLines(context, width, height, rng, condition) {
  if (condition.textureCreaseMarks === false) return;
  const crinkle = clamp01(condition.crinkle);
  const dirt = clamp01(condition.dirtiness);
  const count = Math.max(0, condition.creaseCount || 0);
  for (let i = 0; i < count; i += 1) {
    const angle = (condition.creaseAngle || 0) + (rng() - 0.5) * 0.9;
    const offset = (rng() - 0.5) * height * 0.7;
    context.save();
    context.translate(width / 2, height / 2 + offset);
    context.rotate(angle);
    context.lineCap = "round";
    if (dirt > 0.18 || crinkle > 0.24) {
      context.lineWidth = Math.max(2, height * (0.010 + dirt * 0.020 + crinkle * 0.006));
      context.strokeStyle = "rgba(54,43,30," + Math.min(0.18, dirt * 0.12 + crinkle * 0.04) + ")";
      context.beginPath();
      context.moveTo(-width * 0.64, 0);
      context.lineTo(width * 0.64, 0);
      context.stroke();
    }
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

function makeConditionedNoteTexture(sourceTexture, condition, printTone) {
  const hasPrintTone = Boolean(printTone && printTone.policy !== "off");
  if (!condition && !hasPrintTone) return sourceTexture;
  condition = condition || {};
  const dirt = clamp01(condition.dirtiness);
  const crinkle = clamp01(condition.crinkle);
  const wet = clamp01(condition.wetness);
  const edgeWear = clamp01(condition.edgeWear);
  const textureCreaseMarks = condition.textureCreaseMarks !== false;
  const hasConditionTone = dirt >= 0.01 || crinkle >= 0.01 || wet >= 0.01 || edgeWear >= 0.01;
  if (!hasConditionTone && !hasPrintTone) return sourceTexture;

  const source = sourceTexture.image;
  const sourceWidth = source.naturalWidth || source.width;
  const sourceHeight = source.naturalHeight || source.height;
  const textureResolve = sceneConfig.textureResolve || { policy: "source_exact", maxWidth: 0, blurPx: 0 };
  const maxWidth = Math.max(0, Number(textureResolve.maxWidth || 0));
  const resizeScale = maxWidth > 0 && sourceWidth > maxWidth ? maxWidth / sourceWidth : 1.0;
  const width = Math.max(1, Math.round(sourceWidth * resizeScale));
  const height = Math.max(1, Math.round(sourceHeight * resizeScale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(source, 0, 0, width, height);
  const textureBlurPx = Math.max(0, Number(textureResolve.blurPx || 0));
  if (textureBlurPx > 0) {
    const resolved = document.createElement("canvas");
    resolved.width = width;
    resolved.height = height;
    const resolvedContext = resolved.getContext("2d");
    resolvedContext.drawImage(canvas, 0, 0);
    context.clearRect(0, 0, width, height);
    context.filter = "blur(" + textureBlurPx + "px)";
    context.drawImage(resolved, 0, 0);
    context.filter = "none";
  }
  const rng = browserMulberry32(condition.seed || 1);

  if (hasConditionTone) {
    const wash = applyMaskedOverlay(canvas, (overlay, w, h) => {
      overlay.fillStyle = "rgba(102,84,55," + (dirt * 0.18) + ")";
      overlay.fillRect(0, 0, w, h);
      overlay.fillStyle = "rgba(71,68,58," + (dirt * 0.13) + ")";
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
        const radius = (0.055 + rng() * 0.16) * Math.min(w, h);
        const gradient = overlay.createRadialGradient(x, y, radius * 0.1, x, y, radius);
        gradient.addColorStop(0, "rgba(67,48,28," + (0.10 + dirt * 0.20) + ")");
        gradient.addColorStop(1, "rgba(67,48,28,0)");
        overlay.fillStyle = gradient;
        overlay.fillRect(x - radius, y - radius, radius * 2, radius * 2);
      }
      const grimeBands = textureCreaseMarks
        ? Math.min(6, Math.max(0, Math.round(dirt * 4 + crinkle * 2)))
        : Math.min(4, Math.max(0, Math.round(dirt * 4)));
      for (let i = 0; i < grimeBands; i += 1) {
        const horizontal = rng() < 0.55;
        const x = rng() * w;
        const y = rng() * h;
        overlay.save();
        overlay.translate(x, y);
        overlay.rotate((horizontal ? 0 : Math.PI / 2) + (rng() - 0.5) * 0.5);
        overlay.lineCap = "round";
        overlay.lineWidth = Math.max(2, Math.min(w, h) * (0.012 + rng() * 0.030));
        overlay.strokeStyle = "rgba(48,42,31," + Math.min(0.18, 0.025 + dirt * (0.08 + rng() * 0.05)) + ")";
        overlay.beginPath();
        overlay.moveTo(-w * (0.18 + rng() * 0.24), 0);
        overlay.bezierCurveTo(-w * 0.10, (rng() - 0.5) * h * 0.06, w * 0.10, (rng() - 0.5) * h * 0.08, w * (0.18 + rng() * 0.24), 0);
        overlay.stroke();
        overlay.restore();
      }
      if (textureCreaseMarks && (dirt > 0.20 || crinkle > 0.24)) {
        const foldChance = Math.min(0.92, 0.18 + dirt + crinkle * 0.45);
        if (rng() < foldChance) {
          const foldX = w * (0.42 + (0.58 - 0.42) * rng());
          const foldWidth = w * (0.025 + (0.060 - 0.025) * rng());
          const fold = overlay.createLinearGradient(foldX - foldWidth, 0, foldX + foldWidth, 0);
          fold.addColorStop(0, "rgba(65,54,39,0)");
          fold.addColorStop(0.48, "rgba(55,44,31," + Math.min(0.20, 0.035 + dirt * 0.16 + crinkle * 0.04) + ")");
          fold.addColorStop(0.60, "rgba(246,236,205," + Math.min(0.10, 0.02 + crinkle * 0.07) + ")");
          fold.addColorStop(1, "rgba(65,54,39,0)");
          overlay.fillStyle = fold;
          overlay.fillRect(foldX - foldWidth, 0, foldWidth * 2, h);
        }
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
        overlay.lineWidth = Math.max(1, Math.min(w, h) * (0.006 + edgeWear * 0.012));
        overlay.strokeStyle = "rgba(48,39,28," + Math.min(0.16, dirt * 0.10 + edgeWear * 0.06) + ")";
        for (let i = 0; i < 2; i += 1) {
          const inset = i * overlay.lineWidth * 1.6;
          overlay.strokeRect(inset, inset, w - inset * 2, h - inset * 2);
        }
        overlay.lineWidth = Math.max(1, Math.min(w, h) * (0.004 + edgeWear * 0.010));
        overlay.strokeStyle = "rgba(246,238,211," + (edgeWear * 0.12) + ")";
        for (let i = 0; i < 3; i += 1) {
          const inset = i * overlay.lineWidth * 1.8;
          overlay.strokeRect(inset, inset, w - inset * 2, h - inset * 2);
        }
      }
    });
    context.drawImage(marks, 0, 0);
  }

  applyPrintToneAdjustment(canvas, printTone);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

async function addNotes() {
  for (const asset of [...assets].sort((a, b) => a.layer - b.layer)) {
    const sourceTexture = await loader.loadAsync(asset.textureUrl);
    const texture = makeConditionedNoteTexture(sourceTexture, asset.condition, asset.printTone);
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
    const textureQa = Boolean(sceneConfig.textureQa || asset.textureQa);
    const textureQaEffects = sceneConfig.textureQaEffects || "none";
    const ablation = sceneConfig.appearanceAblation || "full";
    const ablationDisableBacking = ["source_flat", "material_no_backing", "standard_only"].includes(ablation);
    const ablationDisableShadows = ["source_flat", "material_no_shadow", "material_no_backing", "standard_only"].includes(ablation);
    const textureQaFlat = (textureQa && textureQaEffects === "flat") || sceneConfig.appearanceAblation === "source_flat";
    const textureQaBacking = textureQa && ["backing_plane", "postprocess", "condition"].includes(textureQaEffects);
    const materialRoughness = Math.max(
      0.24,
      Math.min(0.96, (asset.roughness ?? 0.82) + (condition.dirtiness || 0) * 0.04 - (condition.wetness || 0) * 0.52)
    );
    if (!ablationDisableBacking && (!textureQa || textureQaBacking)) {
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
    }

    const material = textureQaFlat
      ? new THREE.MeshBasicMaterial({
        map: texture,
        side: THREE.DoubleSide,
        depthTest: false,
        depthWrite: false
      })
      : new THREE.MeshStandardMaterial({
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
    mesh.castShadow = !textureQa && !ablationDisableShadows && asset.castShadow !== false;
    mesh.receiveShadow = false;
    mesh.userData = { material, idColor: asset.idColor, asset };
    meshes.push(mesh);
    scene.add(mesh);
  }
}

function addOccluders() {
  let unknownPropLabelIndex = 0;
  const reservedIdColorKeys = new Set(["0,0,0", ...assets.map((asset) => asset.idColor.join(","))]);
  for (const occluder of occluders) {
    const geometry = makeOccluderGeometry(occluder);
    const texture = makeOccluderTexture(occluder);
    const material = new THREE.MeshStandardMaterial({
      color: texture ? 0xffffff : occluder.color,
      map: texture || null,
      roughness: occluder.textureStyle === "phone" ? 0.50 : 0.88,
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
    const labelAsUnknownProp = NEGATIVE_PROP_LABEL_POLICY === "unknown_banknote" && occluder.propKind === "unknown_banknote";
    const idColor = labelAsUnknownProp
      ? nextUnknownPropIdColor(unknownPropLabelIndex, reservedIdColorKeys)
      : [0, 0, 0];
    if (labelAsUnknownProp) unknownPropLabelIndex += 1;
    mesh.userData = {
      material,
      occluder,
      idColor,
      labelAsUnknownProp,
      labelClassIndex: UNKNOWN_PROP_CLASS_INDEX,
      labelClassName: UNKNOWN_PROP_CLASS_NAME,
      finalColor: idColor,
      auditColor: labelAsUnknownProp ? idColor : [255, 255, 255],
    };
    occluderMeshes.push(mesh);
    scene.add(mesh);
  }
}

await addNotes();
addOccluders();

window.renderPass = (mode) => {
  if (mode === "id") {
    idRenderer.setClearColor(0x000000, 1);
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
      const [r, g, b] = mesh.userData.labelAsUnknownProp ? mesh.userData.idColor : [0, 0, 0];
      mesh.material = new THREE.MeshBasicMaterial({
        color: idColorFromBytes([r, g, b]),
        depthTest: false,
        depthWrite: false
      });
    }
  } else {
    visualRenderer.setClearColor(sceneConfig.surface.scene, 1);
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
  visualRenderer.domElement.style.filter = mode === "id" ? "none" : visualFilterString();
  const activeRenderer = mode === "id" ? idRenderer : visualRenderer;
  activeRenderer.render(scene, camera);
};

function visualFilterString() {
  return [
    "contrast(" + sceneConfig.postprocess.contrast + ")",
    "saturate(" + sceneConfig.postprocess.saturation + ")",
    "brightness(" + sceneConfig.postprocess.brightness + ")",
    sceneConfig.postprocess.focusBlurPx > 0 ? "blur(" + sceneConfig.postprocess.focusBlurPx + "px)" : "",
  ].filter(Boolean).join(" ");
}

function rawCanvasPixels(canvas, filter = "none", overlayCanvas = null, targetWidth = canvas.width, targetHeight = canvas.height) {
  const scratch = document.createElement("canvas");
  scratch.width = targetWidth;
  scratch.height = targetHeight;
  const context = scratch.getContext("2d", { willReadFrequently: true });
  context.filter = filter;
  context.drawImage(canvas, 0, 0, targetWidth, targetHeight);
  context.filter = "none";
  if (overlayCanvas) context.drawImage(overlayCanvas, 0, 0, targetWidth, targetHeight);
  return context.getImageData(0, 0, scratch.width, scratch.height);
}

function samplePixelNearest(source, width, height, x, y, out, offset) {
  const sx = Math.round(x);
  const sy = Math.round(y);
  if (sx < 0 || sy < 0 || sx >= width || sy >= height) {
    out[offset] = 0;
    out[offset + 1] = 0;
    out[offset + 2] = 0;
    out[offset + 3] = 255;
    return;
  }
  const sourceOffset = (sy * width + sx) * 4;
  out[offset] = source[sourceOffset];
  out[offset + 1] = source[sourceOffset + 1];
  out[offset + 2] = source[sourceOffset + 2];
  out[offset + 3] = source[sourceOffset + 3];
}

function samplePixelBilinear(source, width, height, x, y, out, offset) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = x0 + 1;
  const y1 = y0 + 1;
  if (x0 < 0 || y0 < 0 || x1 >= width || y1 >= height) {
    samplePixelNearest(source, width, height, x, y, out, offset);
    return;
  }
  const wx = x - x0;
  const wy = y - y0;
  const weights = [
    [(1 - wx) * (1 - wy), x0, y0],
    [wx * (1 - wy), x1, y0],
    [(1 - wx) * wy, x0, y1],
    [wx * wy, x1, y1],
  ];
  for (let channel = 0; channel < 4; channel += 1) {
    let value = 0;
    for (const [weight, sx, sy] of weights) {
      value += source[(sy * width + sx) * 4 + channel] * weight;
    }
    out[offset + channel] = Math.max(0, Math.min(255, Math.round(value)));
  }
}

function applySharedGeometricTransform(imageData, sampling) {
  const lens = sceneConfig.camera.lensDistortion;
  if (!lens || !lens.applied || Math.abs(lens.k1) < 1e-9) return imageData;
  const { width, height, data } = imageData;
  const output = new ImageData(width, height);
  const out = output.data;
  const k1 = lens.k1;
  const [centerOffsetX, centerOffsetY] = lens.centerOffset || [0, 0];
  const frameScale = lens.frameScale || 1;
  for (let y = 0; y < height; y += 1) {
    const ny = (((y + 0.5) / height) - 0.5 - centerOffsetY) * 2 * frameScale;
    for (let x = 0; x < width; x += 1) {
      const nx = (((x + 0.5) / width) - 0.5 - centerOffsetX) * 2 * frameScale;
      const r2 = nx * nx + ny * ny;
      const factor = Math.max(0.65, 1 + k1 * r2);
      const sx = (((nx / factor) / 2) + 0.5 + centerOffsetX) * width - 0.5;
      const sy = (((ny / factor) / 2) + 0.5 + centerOffsetY) * height - 0.5;
      const offset = (y * width + x) * 4;
      if (sampling === "bilinear") {
        samplePixelBilinear(data, width, height, sx, sy, out, offset);
      } else {
        samplePixelNearest(data, width, height, sx, sy, out, offset);
      }
    }
  }
  return output;
}

function imageDataToPngDataUrl(imageData) {
  const canvas = document.createElement("canvas");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext("2d").putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

function captureCanvasPixels(sourceRenderer = idRenderer, sampling = "nearest") {
  return applySharedGeometricTransform(rawCanvasPixels(sourceRenderer.domElement), sampling);
}

function captureVisualPixels() {
  const imageData = rawCanvasPixels(visualRenderer.domElement, visualFilterString(), grainCanvas, WIDTH, HEIGHT);
  return applySharedGeometricTransform(imageData, "bilinear");
}

function pixelKey(data, offset) {
  return data[offset] + "," + data[offset + 1] + "," + data[offset + 2];
}

function idColorFromBytes([r, g, b]) {
  const color = new THREE.Color();
  color.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
  return color;
}

function nextUnknownPropIdColor(index, reservedIdColorKeys) {
  const side = UNKNOWN_PROP_ID_CHANNELS.length;
  const paletteSize = side * side * side;
  for (let offset = 0; offset < paletteSize; offset += 1) {
    const cursor = (index + offset) % paletteSize;
    const color = [
      UNKNOWN_PROP_ID_CHANNELS[cursor % side],
      UNKNOWN_PROP_ID_CHANNELS[Math.floor(cursor / side) % side],
      UNKNOWN_PROP_ID_CHANNELS[Math.floor(cursor / (side * side)) % side],
    ];
    const key = color.join(",");
    if (!reservedIdColorKeys.has(key)) {
      reservedIdColorKeys.add(key);
      return color;
    }
  }
  throw new Error("No unused id color remains for UNKNOWN_FOREIGN_NOTE props");
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
  for (const mesh of occluderMeshes) {
    if (!mesh.userData.labelAsUnknownProp) continue;
    colorToClass.set(mesh.userData.idColor.join(","), {
      classIndex: mesh.userData.labelClassIndex,
      className: mesh.userData.labelClassName,
      labelSource: "negative_prop_unknown_banknote",
      propKind: mesh.userData.occluder.propKind,
      negativeConfusionHardness: mesh.userData.occluder.negativeConfusionHardness || "none",
    });
  }
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
window.captureVisualPng = () => imageDataToPngDataUrl(captureVisualPixels());
window.captureIdPng = () => imageDataToPngDataUrl(captureCanvasPixels(idRenderer, "nearest"));
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

async function renderStateWithPage(page, renderState, browserMode) {
  fs.mkdirSync(renderState.outDir, { recursive: true });
  await page.setViewport({ width: WIDTH, height: HEIGHT, deviceScaleFactor: 1 });
  const htmlPath = path.join(renderState.outDir, "smoke.html");
  fs.writeFileSync(htmlPath, html(renderState));
  await page.goto(pathToFileURL(htmlPath).href, { waitUntil: "domcontentloaded", timeout: 180000 });
  await page.waitForFunction("window.__cashsnapReady === true");
  await page.evaluate(() => window.renderPass("visual"));
  const visualPngDataUrl = await page.evaluate(() => window.captureVisualPng());
  writeDataUrlPng(visualPngDataUrl, path.join(renderState.outDir, "visual.png"));
  const idPngDataUrl = await page.evaluate(() => {
    window.renderPass("id");
    return window.captureIdPng();
  });
  writeDataUrlPng(idPngDataUrl, path.join(renderState.outDir, "id.png"));
  const boxes = await page.evaluate(() => window.extractIdBoxes());
  const layerAudit = await page.evaluate(() => window.auditLayerOrder());
  if (layerAudit.violations !== 0) {
    throw new Error(`Layer-order audit failed with ${layerAudit.violations} violating pixels`);
  }
  fs.writeFileSync(path.join(renderState.outDir, "visible_boxes.json"), JSON.stringify({ boxes }, null, 2));
  fs.writeFileSync(path.join(renderState.outDir, "layer_audit.json"), JSON.stringify(layerAudit, null, 2));
  const labelText = boxes
    .map((box) => `${box.classIndex} ${box.yolo.map((value) => Number(value).toFixed(6)).join(" ")}`)
    .join("\n");
  fs.writeFileSync(
    path.join(renderState.outDir, "labels_visible.txt"),
    labelText ? `${labelText}\n` : ""
  );
  fs.writeFileSync(
    path.join(renderState.outDir, "metadata.json"),
    JSON.stringify({
      renderer: "three-webgl-edge",
      browserExecutable: BROWSER_EXECUTABLE,
      browserMode,
      variant: renderState.variant,
      sceneMode: renderState.effectiveSceneMode,
      width: WIDTH,
      height: HEIGHT,
      visualScale: VISUAL_SCALE,
      minVisiblePixels: MIN_VISIBLE_PIXELS,
      occluderPolicy: OCCLUDER_POLICY,
      negativePropPolicy: NEGATIVE_PROP_POLICY,
      sceneDefaultOccluderCount: renderState.sceneDefaultOccluders.length,
      sceneConfig: renderState.config,
      assetSelection: {
        sidePolicy: ASSET_SIDE_POLICY,
        stackPosePolicy: STACK_POSE_POLICY,
        cleanOrientationPolicy: CLEAN_ORIENTATION_POLICY,
        classSequence: CLASS_SEQUENCE,
        sideCounts: renderState.assetSideCounts,
        frontBackMixSatisfied: renderState.frontBackMixSatisfied,
      },
      visibilityModel: "explicit-layer-order",
      noteDepthPolicy: "banknote planes use renderOrder with depthTest/depthWrite disabled to avoid impossible surface interpenetration in visible masks",
      assets: renderState.textureAssets,
      occluders: renderState.occluders,
    }, null, 2)
  );
  console.log(`wrote ${renderState.outDir}`);
}

function batchManifestEntries() {
  if (!BATCH_MANIFEST) {
    return [{ variant: VARIANT, outDir: OUT_DIR }];
  }
  const payload = JSON.parse(fs.readFileSync(path.resolve(ROOT, BATCH_MANIFEST), "utf-8"));
  const entries = Array.isArray(payload) ? payload : payload.variants;
  if (!Array.isArray(entries) || entries.length === 0) {
    throw new Error("--batch-manifest must contain a non-empty variants array");
  }
  return entries.map((entry) => ({
    variant: Number.parseInt(String(entry.variant), 10),
    outDir: path.resolve(ROOT, String(entry.outDir)),
  }));
}

async function main() {
  if (!BROWSER_WS_ENDPOINT && !fs.existsSync(BROWSER_EXECUTABLE)) {
    throw new Error(`Browser executable not found at ${BROWSER_EXECUTABLE}`);
  }
  const entries = batchManifestEntries();
  const browser = BROWSER_WS_ENDPOINT
    ? await puppeteer.connect({ browserWSEndpoint: BROWSER_WS_ENDPOINT })
    : await puppeteer.launch({
      executablePath: BROWSER_EXECUTABLE,
      headless: "new",
      args: [
        "--allow-file-access-from-files",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--enable-gpu-rasterization",
        "--ignore-gpu-blocklist",
        "--use-angle=d3d11",
      ],
    });
  let page = null;
  try {
    page = await browser.newPage();
    page.on("console", (message) => console.log(`[browser:${message.type()}] ${message.text()}`));
    page.on("pageerror", (error) => console.error(`[browser:pageerror] ${error.message}`));
    page.setDefaultTimeout(180000);
    page.setDefaultNavigationTimeout(180000);
    const browserMode = BATCH_MANIFEST
      ? (BROWSER_WS_ENDPOINT ? "connected_shared_batch" : "launched_batch")
      : (BROWSER_WS_ENDPOINT ? "connected_shared" : "launched_per_variant");
    for (const entry of entries) {
      if (!Number.isInteger(entry.variant)) {
        throw new Error(`Invalid batch variant: ${JSON.stringify(entry)}`);
      }
      const renderState = buildRenderState(entry.variant, entry.outDir);
      await renderStateWithPage(page, renderState, browserMode);
    }
  } finally {
    if (page) await page.close().catch(() => {});
    if (BROWSER_WS_ENDPOINT) {
      browser.disconnect();
    } else {
      await browser.close();
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
