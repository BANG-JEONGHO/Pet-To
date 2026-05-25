"""
NanoBanana API client for pet virtual try-on refinement.

Supports two modes:
  - "full": text-to-image / image-to-image try-on from scratch (baseline)
  - "refine": image editing on a pre-generated IDM-VTON result (our pipeline)
"""

import base64
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from PIL import Image


NANOBANANA_API_BASE = "https://nanobananaapi.ai/nanobanana-api"
GENERATE_OR_EDIT_ENDPOINT = f"{NANOBANANA_API_BASE}/generate-or-edit-image"
TASK_DETAILS_ENDPOINT = f"{NANOBANANA_API_BASE}/get-task-details"

DEFAULT_POLL_INTERVAL = 3
DEFAULT_MAX_POLL = 120


@dataclass
class NanoBananaUsage:
    api_calls: int = 0
    images_generated: int = 0
    estimated_cost_usd: float = 0.0
    mode: str = ""
    details: list = field(default_factory=list)

    def add_call(self, action: str, cost_per_image: float, count: int = 1):
        self.api_calls += 1
        self.images_generated += count
        self.estimated_cost_usd += cost_per_image * count
        self.details.append({
            "action": action,
            "count": count,
            "cost": cost_per_image * count,
        })

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "api_calls": self.api_calls,
            "images_generated": self.images_generated,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "details": self.details,
        }


def _get_api_key() -> str:
    key = os.environ.get("NANOBANANA_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "NANOBANANA_API_KEY environment variable is not set. "
            "Get your key from https://nanobananaapi.ai/"
        )
    return key


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }


def _image_to_data_url(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _upload_and_get_url(image_path: str) -> str:
    """
    NanoBanana API requires publicly accessible URLs.
    If the image is local, convert to base64 data URL.
    Some API providers accept data URLs; if not, the user should
    host images on a CDN or use a presigned URL.
    """
    path = Path(image_path)
    if path.exists():
        return _image_to_data_url(str(path))
    return image_path


def _poll_task(task_id: str, poll_interval: int = DEFAULT_POLL_INTERVAL,
               max_wait: int = DEFAULT_MAX_POLL) -> dict:
    elapsed = 0
    while elapsed < max_wait:
        resp = requests.get(
            TASK_DETAILS_ENDPOINT,
            params={"task_id": task_id},
            headers=_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

        status = result.get("status", "")
        if status == "completed":
            return result
        if status in ("failed", "error"):
            raise RuntimeError(f"NanoBanana task failed: {result}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(f"NanoBanana task {task_id} did not complete within {max_wait}s")


def _submit_request(payload: dict) -> dict:
    resp = requests.post(
        GENERATE_OR_EDIT_ENDPOINT,
        headers=_headers(),
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _download_result_image(result: dict, output_path: str) -> str:
    data = result.get("data", {})
    images = data if isinstance(data, list) else [data]

    for item in images:
        url = item.get("image_url") or item.get("url", "")
        if url:
            img_resp = requests.get(url, timeout=60)
            img_resp.raise_for_status()
            img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            img.save(output_path, quality=95)
            return output_path

    raise ValueError(f"No image URL found in NanoBanana response: {result}")


# ──────────────────────────────────────────────
#  Mode A: full try-on via NanoBanana (baseline)
# ──────────────────────────────────────────────

COST_PER_IMAGE_EDIT = 0.02
COST_PER_IMAGE_GENERATE = 0.02


def full_tryon_nanobanana(
    pet_image_path: str,
    cloth_image_path: str,
    output_path: str,
    prompt: Optional[str] = None,
    usage: Optional[NanoBananaUsage] = None,
) -> str:
    """
    Baseline: send raw pet + cloth images directly to NanoBanana API.
    No local preprocessing. The API does all the work.
    """
    if usage is None:
        usage = NanoBananaUsage(mode="full_nanobanana")

    if prompt is None:
        prompt = (
            "Make this pet wear the clothing item shown in the second image. "
            "The clothing should naturally fit the pet's body, preserving the "
            "pet's face, legs, and tail. Output a photorealistic result."
        )

    pet_url = _upload_and_get_url(pet_image_path)
    cloth_url = _upload_and_get_url(cloth_image_path)

    payload = {
        "action": "edit",
        "prompt": prompt,
        "image_urls": [pet_url, cloth_url],
        "count": 1,
    }

    result = _submit_request(payload)
    usage.add_call("full_tryon_edit", COST_PER_IMAGE_EDIT, count=1)

    task_id = result.get("task_id")
    if task_id:
        result = _poll_task(task_id)

    _download_result_image(result, output_path)
    return output_path


# ──────────────────────────────────────────────
#  Mode B: refine only (our pipeline output)
# ──────────────────────────────────────────────

def refine_with_nanobanana(
    idm_vton_result_path: str,
    pet_image_path: str,
    cloth_image_path: str,
    output_path: str,
    prompt: Optional[str] = None,
    usage: Optional[NanoBananaUsage] = None,
) -> str:
    """
    Our pipeline: local IDM-VTON delta already generated a try-on image.
    NanoBanana is only used to refine quality (fix artifacts, enhance detail).
    """
    if usage is None:
        usage = NanoBananaUsage(mode="local_pipeline_plus_refine")

    if prompt is None:
        prompt = (
            "This is a photo of a pet wearing clothes. "
            "Refine and enhance this image: fix any visual artifacts, "
            "improve clothing texture and fit, make it look photorealistic. "
            "Keep the pet's pose, face, and overall composition unchanged."
        )

    vton_url = _upload_and_get_url(idm_vton_result_path)

    payload = {
        "action": "edit",
        "prompt": prompt,
        "image_urls": [vton_url],
        "count": 1,
    }

    result = _submit_request(payload)
    usage.add_call("refine_edit", COST_PER_IMAGE_EDIT, count=1)

    task_id = result.get("task_id")
    if task_id:
        result = _poll_task(task_id)

    _download_result_image(result, output_path)
    return output_path


# ──────────────────────────────────────────────
#  Dry-run mode (no actual API call)
# ──────────────────────────────────────────────

def simulate_full_tryon(usage: Optional[NanoBananaUsage] = None) -> NanoBananaUsage:
    """Simulate cost for full NanoBanana try-on (no API call)."""
    if usage is None:
        usage = NanoBananaUsage(mode="full_nanobanana")
    usage.add_call("full_tryon_edit", COST_PER_IMAGE_EDIT, count=1)
    return usage


def simulate_refine(usage: Optional[NanoBananaUsage] = None) -> NanoBananaUsage:
    """Simulate cost for refinement-only NanoBanana call (no API call)."""
    if usage is None:
        usage = NanoBananaUsage(mode="local_pipeline_plus_refine")
    usage.add_call("refine_edit", COST_PER_IMAGE_EDIT, count=1)
    return usage
