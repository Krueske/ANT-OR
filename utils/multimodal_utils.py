"""Shared multimodal helpers extracted from src/*.py.

Provides the image-similarity stack plus the unified multimodal-data helpers.
Implementations are the canonical versions chosen during the consolidation
work: extract_multimodal_data uses the 3-layer fallback strategy from
evaluate_mas.py, format_multimodal_query uses the "text first, double-\\n\\n,
trailing image placeholders" layout used by evaluate_antor / optimize_antor /
optimize_single_spo, and compute_multimodal_similarity uses the cleanest of
the four near-identical variants previously seen across the 6 scripts.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import requests
import base64

from utils.samples_retriever import cosine_similarity, get_or_create_embedding


# Module-level state shared across callers in the same Python process.
_image_feature_cache = {}

# Pre-computed image embeddings (loaded once at import time).
# Path may be overridden via the ANTOR_IMAGE_EMBEDDING_PATH env var (matches
# the behavior of optimize_antor.py and evaluate_single_agent.py); the default
# is the same hardcoded path used by evaluate_mas.py and evaluate_antor.py.
image_embeddings = {}
try:
    image_embedding_path = os.environ.get(
        "ANTOR_IMAGE_EMBEDDING_PATH", "./image_embedding.jsonl"
    )
    if os.path.exists(image_embedding_path):
        with open(image_embedding_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                image_embeddings[data["query"]] = data["embedding"]
        print(f"[Image Similarity] Loaded {len(image_embeddings)} pre-computed image embeddings")
except Exception as e:
    print(f"[Image Similarity] Failed to load pre-computed embeddings: {e}")


def load_image(image_url: str):
    """Load image (supports URL, local path, base64). Returns PIL.Image.Image or None."""
    try:
        from PIL import Image
        import io

        if image_url.startswith('http://') or image_url.startswith('https://'):
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        elif image_url.startswith('data:image'):
            base64_data = image_url.split(',')[1] if ',' in image_url else image_url
            image_data = base64.b64decode(base64_data)
            return Image.open(io.BytesIO(image_data))
        else:
            return Image.open(image_url)
    except Exception as e:
        print(f"[Image Load Failed] {image_url}: {e}")
        return None


def compute_phash_similarity(image1, image2) -> float:
    """Compute perceptual-hash similarity (0..1). Returns 0.5 if imagehash unavailable."""
    try:
        import imagehash

        hash1 = imagehash.phash(image1)
        hash2 = imagehash.phash(image2)
        hamming_distance = hash1 - hash2
        similarity = 1.0 - (hamming_distance / 64.0)
        return max(0.0, similarity)
    except ImportError:
        return 0.5
    except Exception:
        return 0.5


def compute_color_histogram_similarity(image1, image2) -> float:
    """Compute HSV-histogram correlation similarity (0..1). Returns 0.5 if cv2 unavailable."""
    try:
        import cv2
        import numpy as np

        img1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0.0, similarity)
    except ImportError:
        return 0.5
    except Exception:
        return 0.5


def compute_orb_similarity(image1, image2) -> float:
    """Compute ORB feature-match similarity (0..1). Returns 0.5 if cv2 unavailable."""
    try:
        import cv2
        import numpy as np

        img1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) == 0:
            return 0.0

        good_matches = [m for m in matches if m.distance < 50]
        if len(good_matches) == 0:
            return 0.0

        similarity = min(1.0, len(good_matches) / min(len(kp1), len(kp2)))
        return similarity
    except ImportError:
        return 0.5
    except Exception:
        return 0.5


def compute_image_similarity_traditional(image_url1: str, image_url2: str) -> float:
    """Composite traditional-feature image similarity (phash 0.4 + color 0.35 + orb 0.25)."""
    cache_key = (image_url1, image_url2)
    reverse_key = (image_url2, image_url1)
    if cache_key in _image_feature_cache:
        return _image_feature_cache[cache_key]
    if reverse_key in _image_feature_cache:
        return _image_feature_cache[reverse_key]

    img1 = load_image(image_url1)
    img2 = load_image(image_url2)

    if img1 is None or img2 is None:
        return 0.0

    from PIL import Image
    target_size = (256, 256)
    img1 = img1.resize(target_size, Image.Resampling.LANCZOS)
    img2 = img2.resize(target_size, Image.Resampling.LANCZOS)

    similarities = []
    weights = []

    phash_sim = compute_phash_similarity(img1, img2)
    similarities.append(phash_sim)
    weights.append(0.4)

    color_sim = compute_color_histogram_similarity(img1, img2)
    similarities.append(color_sim)
    weights.append(0.35)

    orb_sim = compute_orb_similarity(img1, img2)
    similarities.append(orb_sim)
    weights.append(0.25)

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.5

    final_similarity = sum(s * w for s, w in zip(similarities, weights)) / total_weight
    _image_feature_cache[cache_key] = final_similarity

    return final_similarity


def compute_image_similarity(image_url1: str, image_url2: str) -> float:
    """Image similarity: prefer pre-computed embeddings, fall back to traditional features."""
    embedding1 = image_embeddings.get(image_url1)
    embedding2 = image_embeddings.get(image_url2)
    if embedding1 and embedding2:
        return cosine_similarity(embedding1, embedding2)
    else:
        return compute_image_similarity_traditional(image_url1, image_url2)


# =============================================================================
# Multimodal data helpers (unified across scripts)
# =============================================================================

# Default weights for combining text and image similarity.
TEXT_SIMILARITY_WEIGHT = 0.5
IMAGE_SIMILARITY_WEIGHT = 0.5

# Recognized keys when probing dicts for text or image content.
_TEXT_KEYS_PRIORITY = ("query_text", "query", "text", "content", "input")
_IMAGE_KEYS = ("image_urls", "images", "image_url", "image", "img_urls", "imgs")


def extract_multimodal_data(query_dict) -> Tuple[str, List[str]]:
    """Extract (text, image_urls) from a query value.

    Strategy (canonicalized from evaluate_mas.py — 3-layer fallback):
      1. Look at known text keys (query_text/query/text/content/input).
         - If the value is a string, use it directly.
         - If the value is a nested dict with a "text" key, take that.
         - Otherwise json.dumps the nested dict's non-image keys.
      2. After image extraction, if no text was found, try the top-level "text" key.
      3. Final fallback: json.dumps the top-level dict's non-image keys.

    Strings pass through unchanged. Non-dict, non-string inputs are str()-coerced.
    """
    if isinstance(query_dict, str):
        return query_dict, []
    if not isinstance(query_dict, dict):
        return str(query_dict), []

    text = ""
    for key in _TEXT_KEYS_PRIORITY:
        if key in query_dict:
            value = query_dict[key]
            if isinstance(value, str):
                text = value
                break
            elif isinstance(value, dict):
                if "text" in value:
                    text = value["text"]
                    break
                else:
                    text_parts = {
                        k: v for k, v in value.items() if k not in _IMAGE_KEYS
                    }
                    if text_parts:
                        text = json.dumps(text_parts, ensure_ascii=False)
                        break

    images: List[str] = []
    for key in ("query", "query_dict"):
        if key in query_dict and isinstance(query_dict[key], dict):
            nested = query_dict[key]
            for img_key in _IMAGE_KEYS:
                if img_key in nested:
                    value = nested[img_key]
                    if isinstance(value, list):
                        images = value
                    elif isinstance(value, str):
                        images = [value]
                    break
            if images:
                break

    if not images:
        for key in _IMAGE_KEYS:
            if key in query_dict:
                value = query_dict[key]
                if isinstance(value, list):
                    images = value
                elif isinstance(value, str):
                    images = [value]
                break

    if not text:
        if "text" in query_dict:
            text = query_dict["text"]
        else:
            text_parts = {
                k: v for k, v in query_dict.items() if k not in _IMAGE_KEYS
            }
            if text_parts:
                text = json.dumps(text_parts, ensure_ascii=False)

    return text, images


def format_multimodal_query(query_text: str, image_urls: List[str]) -> str:
    """Append <image> placeholders after the query text using double-\\n\\n separators.

    Layout (text first, trailing placeholders, double-\\n\\n separators) is the
    most common variant across the existing scripts (evaluate_antor /
    optimize_antor / optimize_single_spo).

    Idempotent: if the query_text already contains at least len(image_urls)
    placeholders, returns it unchanged.
    """
    if not image_urls:
        return query_text
    existing = query_text.count("<image>")
    if existing >= len(image_urls) and "<image>" in query_text:
        return query_text
    placeholders = "\n\n".join(["<image>" for _ in image_urls])
    return f"{query_text}\n\n{placeholders}"


def compute_multimodal_similarity(query1: Dict, query2: Dict) -> float:
    """Combined text+image similarity for two query dicts.

    Returns text-only similarity when neither side has images. Penalizes the
    one-side-has-images case via the asymmetric branch. Otherwise returns the
    weighted combination of text and image similarity.
    """
    text1, images1 = extract_multimodal_data(query1)
    text2, images2 = extract_multimodal_data(query2)

    text_sim = 0.0
    if text1 and text2:
        emb1 = get_or_create_embedding(text1)
        emb2 = get_or_create_embedding(text2)
        if emb1 is not None and emb2 is not None and emb1 != 0 and emb2 != 0:
            text_sim = cosine_similarity(emb1, emb2)

    image_sim = 0.0
    if images1 and images2:
        sim_sum = 0.0
        count = 0
        for img1 in images1:
            for img2 in images2:
                sim_sum += compute_image_similarity(img1, img2)
                count += 1
        if count > 0:
            image_sim = sim_sum / count
    elif images1 or images2:
        image_sim = 0.0
    else:
        image_sim = 0.5

    if not images1 and not images2:
        return text_sim

    if (images1 and not images2) or (images2 and not images1):
        return TEXT_SIMILARITY_WEIGHT * text_sim + (1 - TEXT_SIMILARITY_WEIGHT) * image_sim * 0.5

    return TEXT_SIMILARITY_WEIGHT * text_sim + IMAGE_SIMILARITY_WEIGHT * image_sim


# =============================================================================
# Image I/O + LLM-message construction (unified across scripts)
# =============================================================================

def encode_image_to_base64(image_url: str) -> Optional[str]:
    """Return base64 string for an image source. Returns None on failure.

    Handles three input forms:
      - data:image:* URI → strips the prefix and returns the inline base64
      - http(s):// URL → fetches and base64-encodes the response body
      - local path → reads the file and base64-encodes its bytes
    """
    if image_url.startswith('data:image'):
        return image_url.split(',')[1] if ',' in image_url else image_url
    try:
        if image_url.startswith('http://') or image_url.startswith('https://'):
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image_data = response.content
        else:
            with open(image_url, 'rb') as f:
                image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"Image encoding failed: {e}")
        return None


def get_image_mime_type(image_url: str) -> str:
    """Return MIME type by URL/path extension. Defaults to image/jpeg for unknown types."""
    lower = image_url.lower()
    if lower.endswith('.png'):
        return 'image/png'
    elif lower.endswith('.gif'):
        return 'image/gif'
    elif lower.endswith(('.jpg', '.jpeg')):
        return 'image/jpeg'
    elif lower.endswith('.webp'):
        return 'image/webp'
    elif lower.endswith('.bmp'):
        return 'image/bmp'
    else:
        return 'image/jpeg'


def build_image_content(image_url: str) -> Optional[Dict]:
    """Build an OpenAI-compatible image content block.

    URL and data:image: inputs pass through unchanged; local paths are
    base64-encoded and wrapped as a data URI. Returns None for empty input
    or if local file encoding fails.
    """
    if not image_url:
        return None
    if image_url.startswith('http://') or image_url.startswith('https://'):
        return {"type": "image_url", "image_url": {"url": image_url}}
    elif image_url.startswith('data:image'):
        return {"type": "image_url", "image_url": {"url": image_url}}
    else:
        base64_data = encode_image_to_base64(image_url)
        if base64_data:
            mime_type = get_image_mime_type(image_url)
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
            }
        return None


def replace_image_placeholders(prompt: str, image_urls: List[str]) -> List[Dict]:
    """Interleave <image> placeholders in `prompt` with concrete image content blocks.

    Uses re.split keeping the delimiter so order between text segments and
    images is preserved exactly. Extra image_urls (beyond the placeholder count)
    are silently dropped; extra placeholders simply produce no image.
    """
    content: List[Dict] = []
    parts = re.split(r'(<image>)', prompt)
    image_idx = 0
    current_text = ""
    for part in parts:
        if part == "<image>":
            if current_text:
                content.append({"type": "text", "text": current_text})
                current_text = ""
            if image_idx < len(image_urls):
                image_content = build_image_content(image_urls[image_idx])
                if image_content:
                    content.append(image_content)
                image_idx += 1
        else:
            current_text += part
    if current_text:
        content.append({"type": "text", "text": current_text})
    return content


def prompt_to_messages_multimodal(prompt: str, image_urls: List[str] = None) -> List[Dict]:
    """Build OpenAI chat messages with text + images.

    If the prompt contains <image> placeholders, image_urls are interleaved
    via replace_image_placeholders. Otherwise the prompt text is appended
    first and the images are appended afterwards, ensuring images are never
    silently dropped.
    """
    content: List[Dict] = []
    if image_urls and "<image>" in prompt:
        content = replace_image_placeholders(prompt, image_urls)
    else:
        if prompt:
            content.append({"type": "text", "text": prompt})
        if image_urls:
            for image_url in image_urls:
                image_content = build_image_content(image_url)
                if image_content:
                    content.append(image_content)
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]


def prompt_to_messages(prompt: str, image_urls: List[str] = None) -> List[Dict]:
    """Build OpenAI chat messages. Delegates to multimodal path when images are present."""
    if image_urls:
        return prompt_to_messages_multimodal(prompt, image_urls)
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
