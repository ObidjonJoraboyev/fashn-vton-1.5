"""Post-processing: resolution restore, background/identity compositing, sharpening, sample ranking.

Some enhancers below (face-identity scoring, Real-ESRGAN super-resolution, GFPGAN
face restoration) depend on the optional `enhance` extra (`pip install fashn-vton[enhance]`).
They are imported lazily and cached; if unavailable or if loading fails for any
reason, every function here falls back to its built-in behavior instead of raising,
so a base install is never affected.
"""

import functools
import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

_logger = logging.getLogger("fashn_vton.postprocess")


def resize_to_match(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Resize a PIL image to (width, height) using high-quality Lanczos resampling."""
    if image.size == target_size:
        return image
    return image.resize(target_size, Image.LANCZOS)


@functools.lru_cache(maxsize=4)
def _get_upsampler(scale: int):
    """Lazily load a Real-ESRGAN upsampler for the given integer scale. None if unavailable."""
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        model_name = "RealESRGAN_x4plus" if scale == 4 else "RealESRGAN_x2plus"
        return RealESRGANer(
            scale=scale,
            model_path=f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/{model_name}.pth",
            model=model,
            half=False,
        )
    except Exception as e:
        _logger.warning(
            f"Super-resolution unavailable ({e}); falling back to Lanczos resize. "
            "Install with `pip install fashn-vton[enhance]` for sharper upscaling."
        )
        return None


def smart_resize(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Resize to `target_size`. When upscaling and Real-ESRGAN is available, runs it
    first to add real detail before the final resize, instead of just interpolating
    with Lanczos. Falls back to plain Lanczos resize otherwise (same as `resize_to_match`).
    """
    if image.size == target_size:
        return image

    is_upscale = target_size[0] * target_size[1] > image.size[0] * image.size[1]
    if not is_upscale:
        return resize_to_match(image, target_size)

    scale = 4 if max(target_size) / max(image.size) > 2 else 2
    upsampler = _get_upsampler(scale=scale)
    if upsampler is None:
        return resize_to_match(image, target_size)

    try:
        img_bgr = np.array(image.convert("RGB"))[..., ::-1]
        output, _ = upsampler.enhance(img_bgr)
        upscaled = Image.fromarray(output[..., ::-1].copy())
        return resize_to_match(upscaled, target_size)
    except Exception as e:
        _logger.warning(f"Real-ESRGAN upscale failed ({e}); falling back to Lanczos resize.")
        return resize_to_match(image, target_size)


@functools.lru_cache(maxsize=1)
def _get_face_restorer():
    """Lazily load GFPGAN. None if unavailable."""
    try:
        from gfpgan import GFPGANer

        return GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
        )
    except Exception as e:
        _logger.warning(
            f"Face restoration unavailable ({e}); install with `pip install fashn-vton[enhance]`."
        )
        return None


def restore_face(image: Image.Image) -> Image.Image:
    """
    Run GFPGAN face restoration over the image (it detects and restores faces
    internally; non-face regions pass through untouched). No-ops if GFPGAN isn't
    installed or restoration fails.

    Note: GFPGAN can subtly alter facial features in pursuit of "enhancement" — it
    trades perfect identity fidelity for sharper detail. Most useful when running
    with preserve_background=False, since with preserve_background=True the face
    region is already restored to the exact original pixels during compositing.
    """
    restorer = _get_face_restorer()
    if restorer is None:
        return image
    try:
        img_bgr = np.array(image.convert("RGB"))[..., ::-1]
        _, _, restored_bgr = restorer.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
        return Image.fromarray(restored_bgr[..., ::-1].copy())
    except Exception as e:
        _logger.warning(f"GFPGAN restoration failed ({e}); returning unrestored image.")
        return image


@functools.lru_cache(maxsize=1)
def _get_face_analyzer():
    """Lazily load InsightFace's ArcFace-based face analyzer. None if unavailable."""
    try:
        import insightface

        app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(320, 320))
        return app
    except Exception as e:
        _logger.warning(
            f"Face-identity scoring unavailable ({e}); install with `pip install fashn-vton[enhance]`."
        )
        return None


def face_identity_score(generated: Image.Image, reference: Image.Image) -> Optional[float]:
    """
    Cosine similarity between ArcFace embeddings of the largest detected face in
    `generated` vs `reference`, for ranking best-of-N samples by identity preservation.

    Returns None (callers should ignore it) if InsightFace isn't installed or no
    face is confidently detected in either image. Most informative when running
    with preserve_background=False — with the default preserve_background=True the
    face is already pixel-identical across all samples, so this score won't vary.
    """
    app = _get_face_analyzer()
    if app is None:
        return None

    def _embedding(img: Image.Image):
        bgr = np.array(img.convert("RGB"))[..., ::-1].copy()
        faces = app.get(bgr)
        if not faces:
            return None
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return largest.normed_embedding

    emb_gen, emb_ref = _embedding(generated), _embedding(reference)
    if emb_gen is None or emb_ref is None:
        return None

    return float(np.dot(emb_gen, emb_ref))


def _feather_mask(mask: np.ndarray, feather_px: int) -> np.ndarray:
    """Convert a boolean mask to a soft float alpha mask with feathered edges."""
    alpha = mask.astype(np.float32)
    if feather_px > 0:
        ksize = feather_px * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), feather_px / 2.0)
    return np.clip(alpha, 0.0, 1.0)


def composite_preserve_background(
    generated: Image.Image,
    original: Image.Image,
    edit_mask: np.ndarray,
    feather_px: int = 8,
) -> Image.Image:
    """
    Alpha-composite the generated image's edited region over the pixel-exact original.

    The model regenerates the entire canvas from noise, so nothing is guaranteed
    pixel-identical to the input on its own. `edit_mask` marks the region the model
    was meant to change (garment + surrounding workspace); everywhere else is
    forced back to the original input pixels, guaranteeing the background, face,
    and other untouched regions never drift.

    Args:
        generated: Model output image (any resolution)
        original: Original input image whose pixels should be preserved
        edit_mask: Boolean array marking the region allowed to change. Resized to
            `original`'s resolution automatically if shapes don't match.
        feather_px: Blend radius at the mask boundary to avoid a hard seam

    Returns:
        Composited image at `original`'s resolution
    """
    target_size = original.size  # (W, H)
    generated_resized = smart_resize(generated, target_size)

    if edit_mask.shape[:2] != (target_size[1], target_size[0]):
        edit_mask = cv2.resize(
            edit_mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    alpha = _feather_mask(edit_mask, feather_px)[..., None]

    gen_np = np.array(generated_resized).astype(np.float32)
    orig_np = np.array(original.convert("RGB")).astype(np.float32)
    blended = gen_np * alpha + orig_np * (1.0 - alpha)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def unsharp_mask(image: Image.Image, amount: float = 0.6, radius: float = 1.5, threshold: int = 2) -> Image.Image:
    """
    Classic unsharp-mask sharpening to counter the mild softness typical of
    diffusion-model output. `amount` in ~0.3-1.0 is a safe range; higher looks artificial.
    """
    img = np.array(image).astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    diff = img - blurred
    edge_mask = np.abs(diff) >= threshold
    sharpened = img + amount * diff * edge_mask
    return Image.fromarray(sharpened.clip(0, 255).astype(np.uint8))


def score_garment_fidelity(generated: Image.Image, garment_crop_np: np.ndarray, region_mask: np.ndarray) -> float:
    """
    Cheap heuristic for ranking multiple samples: color-histogram correlation between
    the generated garment region and the source garment crop. Higher is better;
    meant for relative ranking across samples, not as an absolute quality score.

    Args:
        generated: One generated candidate image
        garment_crop_np: The garment image as fed to the model (non-garment pixels
            filled with the standard mask_value=127 gray)
        region_mask: Boolean mask of where the new garment should appear in `generated`
            (the same edit mask used for compositing). Resized automatically if needed.

    Returns:
        Histogram correlation in [-1, 1] (0.0 if either region is too small to score)
    """
    gen_np = np.array(generated.convert("RGB"))

    if region_mask.shape[:2] != gen_np.shape[:2]:
        region_mask = cv2.resize(
            region_mask.astype(np.uint8), (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    garment_mask = np.any(garment_crop_np != 127, axis=-1)

    if region_mask.sum() < 50 or garment_mask.sum() < 50:
        return 0.0

    def _hist(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [32, 32], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    hist_gen = _hist(gen_np, region_mask)
    hist_src = _hist(garment_crop_np, garment_mask)
    return float(cv2.compareHist(hist_gen.astype(np.float32), hist_src.astype(np.float32), cv2.HISTCMP_CORREL))
