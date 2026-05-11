from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image


def summarize_image(image: Image.Image | None) -> dict[str, Any]:
    if image is None:
        return {"provided": False}

    width, height = image.size
    mode = image.mode
    aspect = round(width / height, 2) if height else None
    return {
        "provided": True,
        "width": width,
        "height": height,
        "mode": mode,
        "aspect_ratio": aspect,
        "note": (
            "Image received and routed to multimodal retrieval and the final agent when available."
        ),
    }


def image_to_data_url(
    image: Image.Image | None,
    *,
    max_side: int = 768,
    quality: int = 85,
) -> str | None:
    if image is None:
        return None

    prepared = image.convert("RGB")
    prepared.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    prepared.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
