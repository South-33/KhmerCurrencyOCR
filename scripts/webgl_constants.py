"""Shared WebGL synthetic-pipeline constants."""

from __future__ import annotations


WEBGL_ASSET_SIDE_POLICIES = {"any", "front_only", "back_only", "front_back_mix"}

WEBGL_ASSET_QUALITY_POLICIES = {"latest_design", "all_manifest", "filesystem_all"}

WEBGL_STACK_POSE_POLICIES = {"default", "real_aspect_v1", "real_aspect_v2"}

WEBGL_CLEAN_ORIENTATION_POLICIES = {"default", "real_aspect_v1", "real_aspect_square_v1", "real_aspect_bridge_v1"}

WEBGL_NOTE_CONDITION_POLICIES = {"mixed", "scan_fidelity", "pristine_only", "handled_clean", "handled_3d", "heavy_wear", "wet_stress"}

WEBGL_NOTE_PRINT_TONE_POLICIES = {"off", "local_dynamic_range_v1", "bill_auto_exposure_v1", "real_bridge_print_contrast_v1"}

WEBGL_CAMERA_ISP_POLICIES = {
    "default",
    "phone_dynamic_range_v1",
    "phone_dynamic_range_v2",
    "real_bridge_dynamic_range_v1",
    "mined_fp_dark_v1",
}

WEBGL_TEXTURE_QA_EFFECTS = {"flat", "lit_material", "backing_plane", "postprocess", "condition"}

WEBGL_OCCLUDER_POLICIES = {"scene_default", "no_hand", "none"}

WEBGL_NEGATIVE_PROP_POLICIES = {
    "classic",
    "unknown_currency_soft_v1",
    "unknown_currency_soft_dark_v1",
    "unknown_currency_v1",
    "unknown_currency_fullframe_v1",
    "unknown_currency_fullframe_dark_v1",
}

WEBGL_SCENE_MODES = {
    "auto",
    "clean",
    "clean_single",
    "clean_context",
    "texture_qa",
    "negative",
    "stack",
    "fan",
    "thin_edge",
    "hand_occlusion",
    "qa3",
}

WEBGL_CAMERA_PROFILES = {
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
}
