"""Shared WebGL synthetic-pipeline constants."""

from __future__ import annotations


WEBGL_ASSET_SIDE_POLICIES = {"any", "front_only", "back_only", "front_back_mix"}

WEBGL_CAMERA_PROFILES = {
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
}
