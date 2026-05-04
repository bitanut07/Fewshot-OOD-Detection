# -*- coding: utf-8 -*-
"""Download OpenAI CLIP checkpoints from the same public URLs as glali ``clip_w_local/clip.py``.

Weights are then loaded with ``open_clip`` (architecture ``*-quickgelu`` matches the
official ``.pt`` files). This keeps the rest of the stack on ``open_clip`` (e.g.
``forward_intermediates`` for local features) while using identical files to glali.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
import warnings
from typing import Dict, Optional, Tuple

from tqdm import tqdm

# Same URLs / SHA256-in-path convention as glali/clip_w_local/clip.py
OPENAI_CLIP_CHECKPOINT_URLS: Dict[str, str] = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}

# open_clip model names whose configs match official OpenAI ``.pt`` checkpoints
_OPENCLIP_ARCH_FOR_OPENAI_PT: Dict[str, str] = {
    "RN50": "RN50-quickgelu",
    "RN101": "RN101-quickgelu",
    "RN50x4": "RN50x4-quickgelu",
    "RN50x16": "RN50x16-quickgelu",
    "ViT-B/32": "ViT-B-32-quickgelu",
    "ViT-B/16": "ViT-B-16-quickgelu",
}

# Aliases for ``pretrained`` that trigger Azure download + local path load
OPENAI_PUBLIC_PRETRAINED_TAGS = frozenset(
    {"openai_public", "openai-azure", "openai_official", "openai_azure"}
)


def uses_openai_public_weights(pretrained: str) -> bool:
    return pretrained.strip().lower() in {t.lower() for t in OPENAI_PUBLIC_PRETRAINED_TAGS}


def list_openai_public_backbones() -> Tuple[str, ...]:
    return tuple(OPENAI_CLIP_CHECKPOINT_URLS.keys())


def canonical_openai_checkpoint_key(backbone: str) -> str:
    """Map user backbone string to a key in ``OPENAI_CLIP_CHECKPOINT_URLS``."""
    b = backbone.strip().replace("–", "-")
    for k in OPENAI_CLIP_CHECKPOINT_URLS:
        if b == k:
            return k
    bn = b.replace("/", "-").lower()
    for k in OPENAI_CLIP_CHECKPOINT_URLS:
        if k.replace("/", "-").lower() == bn:
            return k
    raise ValueError(
        f"No OpenAI public (.pt) checkpoint for backbone {backbone!r}. "
        f"Supported: {list(OPENAI_CLIP_CHECKPOINT_URLS)}"
    )


def open_clip_model_name_for_openai_checkpoint(checkpoint_key: str) -> str:
    if checkpoint_key not in _OPENCLIP_ARCH_FOR_OPENAI_PT:
        raise KeyError(checkpoint_key)
    return _OPENCLIP_ARCH_FOR_OPENAI_PT[checkpoint_key]


def download_openai_clip_weights(checkpoint_key: str, root: Optional[str] = None) -> str:
    """Download (or reuse) a ``.pt`` file; return absolute path. Same logic as glali ``_download``."""
    url = OPENAI_CLIP_CHECKPOINT_URLS[checkpoint_key]
    root = root or os.environ.get("CLIP_OPENAI_CACHE") or os.path.expanduser("~/.cache/clip")
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        digest = hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        if digest == expected_sha256:
            return download_target
        warnings.warn(
            f"{download_target} exists but SHA256 does not match; re-downloading.",
            stacklevel=2,
        )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        total = source.info().get("Content-Length")
        total_i = int(total) if total else None
        with tqdm(total=total_i, ncols=80, unit="iB", unit_scale=True, desc=filename) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    digest = hashlib.sha256(open(download_target, "rb").read()).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError("Downloaded model failed SHA256 checksum verification")
    return download_target


def resolve_open_clip_load_args(
    backbone: str,
    pretrained: str,
    weight_cache_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Return ``(open_clip_model_name, pretrained_arg)`` for ``create_model_and_transforms``."""
    if not uses_openai_public_weights(pretrained):
        oc_name = backbone.replace("/", "-")
        return oc_name, pretrained

    ck_key = canonical_openai_checkpoint_key(backbone)
    path = download_openai_clip_weights(ck_key, root=weight_cache_dir)
    oc_name = open_clip_model_name_for_openai_checkpoint(ck_key)
    return oc_name, path
