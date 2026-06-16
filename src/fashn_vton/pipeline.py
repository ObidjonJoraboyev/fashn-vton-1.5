"""TryOn Pipeline."""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
from fashn_human_parser import CATEGORY_TO_BODY_COVERAGE, FashnHumanParser
from PIL import Image
from tqdm.auto import tqdm

from . import postprocess
from .dwpose import DWposeDetector, draw_pose
from .preprocessing import (
    BODY_COVERAGE_TO_FASHN_LABELS,
    FASHN_LABELS_TO_IDS,
    AspectPreserveResize,
    ResizePad,
    compute_clothing_agnostic_mask,
    create_clothing_agnostic_image,
    create_garment_image,
)
from .tryon_mmdit import TryOnModel
from .utils import (
    get_dummy_dw_keypoints,
    get_rf_schedule,
    load_checkpoint,
    normalize_uint8_to_neg1_1,
    numpy_to_torch,
    setup_logger,
    tensor_to_pil,
)


@dataclass
class PipelineOutput:
    """Pipeline output container."""

    images: List[Image.Image]
    # Garment-fidelity ranking scores, same order as `images` (best first).
    # Only populated when num_samples > 1; None for single-sample calls.
    scores: Optional[List[float]] = None
    # Human-readable input/category warnings raised during this call, for a
    # caller-facing UI to surface (e.g. "retake the photo") instead of digging
    # through logs.
    warnings: List[str] = field(default_factory=list)
    # Garment-affected region for the final image, at its resolution. Lets
    # callers do their own compositing (e.g. `run_outfit` protecting earlier passes).
    edit_mask: Optional[np.ndarray] = None


class TryOnPipeline:
    """
    TryOn inference pipeline.

    Args:
        weights_dir: Directory containing model weights (model.safetensors, dwpose/)
        device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        logger: Optional logger instance

    Example:
        pipeline = TryOnPipeline(weights_dir="./weights")
        result = pipeline(person_image, garment_image, category="tops")
    """

    CATEGORY_TO_LABEL = {"tops": 1, "bottoms": 2, "one-pieces": 3}

    def __init__(
        self,
        weights_dir: str,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.weights_dir = os.path.abspath(weights_dir)
        self.logger = logger or setup_logger("TryOnPipeline", level=logging.INFO)

        # Setup device
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.logger.info(f"Using device: {self.device}")

        # Setup inference dtype
        self.inference_dtype = torch.float32
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.inference_dtype = torch.bfloat16
        self.logger.info(f"Using dtype: {self.inference_dtype}")

        # Validate weights exist
        self._validate_weights()

        # Load models
        self._setup_tryon_model()
        self._setup_pose_model()
        self._setup_hp_model()

        # Setup transforms (derived from model input shape)
        h, w = self.tryon_model.input_shape
        max_dim = max(h, w)
        self.pre_resize = AspectPreserveResize(target_size=(max_dim, max_dim), mode="fit", backend="pil")
        self.resize_pad_fn = ResizePad((w, h), backend="opencv")

    def _label_indices_for_category(self, category: str) -> List[int]:
        """Garment-region label indices associated with a try-on category."""
        body_coverage = CATEGORY_TO_BODY_COVERAGE.get(category)
        labels = BODY_COVERAGE_TO_FASHN_LABELS.get(body_coverage, [])
        return [FASHN_LABELS_TO_IDS[label] for label in labels]

    def _check_category_match(
        self,
        garment_seg_pred: np.ndarray,
        category: str,
        min_confidence: float = 0.08,
        strict: bool = False,
    ) -> Optional[str]:
        """
        Sanity-check that the garment image actually looks like the requested category.

        Compares pixel coverage of each category's associated labels in the garment
        image's own segmentation and warns (or raises, if `strict`) when a different
        category is clearly more dominant than the one requested.

        Returns the warning message (also logged), or None if no mismatch found.
        """
        coverage_by_category = {}
        for candidate in self.CATEGORY_TO_LABEL:
            indices = self._label_indices_for_category(candidate)
            if indices:
                coverage_by_category[candidate] = float(np.isin(garment_seg_pred, indices).mean())

        if not coverage_by_category:
            return None

        best_category = max(coverage_by_category, key=coverage_by_category.get)
        best_coverage = coverage_by_category[best_category]

        if best_coverage < min_confidence or best_category == category:
            return None

        msg = (
            f"Garment image looks like '{best_category}' ({best_coverage:.0%} of pixels match) "
            f"but category='{category}' was requested. Results may be degraded — "
            "double check the category or crop the garment image more tightly."
        )
        if strict:
            raise ValueError(msg)
        self.logger.warning(msg)
        return msg

    def _validate_inputs(
        self,
        person_image_np: np.ndarray,
        person_pose: dict,
        person_seg_pred: np.ndarray,
        category: str,
        min_resolution: int = 256,
        min_keypoints: int = 8,
        strict: bool = False,
    ) -> List[str]:
        """
        Catch garbage-in-garbage-out cases before spending a full sampling pass:
        too-low resolution, no detectable person, or a photo that doesn't actually
        show the body region the requested category needs (e.g. a half-body crop
        with category="bottoms").

        Returns the list of warning messages found (also logged).
        """
        warnings: List[str] = []

        h, w = person_image_np.shape[:2]
        if min(h, w) < min_resolution:
            msg = (
                f"person_image is low resolution ({w}x{h}); results may be degraded. "
                f"Recommended minimum: {min_resolution}px on the shorter side."
            )
            self.logger.warning(msg)
            warnings.append(msg)

        subset = person_pose["bodies"]["subset"]
        visible_keypoints = int(np.sum(subset[:, :18] != -1))
        if visible_keypoints == 0:
            msg = "No person detected in person_image (zero visible pose keypoints)."
            if strict:
                raise ValueError(msg)
            self.logger.warning(msg)
            warnings.append(msg)
        elif visible_keypoints < min_keypoints:
            msg = (
                f"Low pose confidence: only {visible_keypoints}/18 body keypoints detected. "
                "Use a clearer, front-facing, well-lit, single-person photo for best results."
            )
            self.logger.warning(msg)
            warnings.append(msg)

        body_coverage = CATEGORY_TO_BODY_COVERAGE.get(category)
        required_labels = {
            "upper": ["torso", "arms"],
            "lower": ["legs"],
            "full": ["torso", "arms", "legs"],
        }.get(body_coverage, [])

        for label in required_labels:
            label_id = FASHN_LABELS_TO_IDS.get(label)
            if label_id is None:
                continue
            coverage = float(np.mean(person_seg_pred == label_id))
            if coverage < 0.01:
                msg = (
                    f"category='{category}' needs a visible '{label}', but person_image shows "
                    f"almost none ({coverage:.1%}). Photo may be cropped too tightly."
                )
                if strict:
                    raise ValueError(msg)
                self.logger.warning(msg)
                warnings.append(msg)

        return warnings

    def _validate_weights(self):
        """Check that required weight files exist."""
        tryon_path = os.path.join(self.weights_dir, "model.safetensors")
        dwpose_dir = os.path.join(self.weights_dir, "dwpose")
        yolox_path = os.path.join(dwpose_dir, "yolox_l.onnx")
        dwpose_path = os.path.join(dwpose_dir, "dw-ll_ucoco_384.onnx")

        missing = []
        if not os.path.exists(tryon_path):
            missing.append(tryon_path)
        if not os.path.exists(yolox_path):
            missing.append(yolox_path)
        if not os.path.exists(dwpose_path):
            missing.append(dwpose_path)

        if missing:
            raise FileNotFoundError(
                "Missing model weights:\n"
                + "\n".join(f"  - {p}" for p in missing)
                + f"\n\nPlease run:\n  python scripts/download_weights.py --weights-dir {self.weights_dir}"
            )

    def _setup_tryon_model(self):
        """Load the TryOn model."""
        model_path = os.path.join(self.weights_dir, "model.safetensors")
        self.logger.info(f"Loading TryOnModel from {model_path}")

        self.tryon_model = TryOnModel()
        state_dict = load_checkpoint(model_path, device=str(self.device))
        self.tryon_model.load_state_dict(state_dict)
        self.tryon_model.to(self.device, dtype=self.inference_dtype).eval()

        self.logger.info("TryOnModel loaded")

    def _setup_pose_model(self):
        """Load DWPose model."""
        dwpose_dir = os.path.join(self.weights_dir, "dwpose")
        self.logger.info(f"Loading DWPose from {dwpose_dir}")

        dwpose_device = f"cuda:{self.device.index or 0}" if self.device.type == "cuda" else "cpu"
        self.pose_model = DWposeDetector(checkpoints_dir=dwpose_dir, device=dwpose_device)

        self.logger.info("DWPose loaded")

    def _setup_hp_model(self):
        """Load human parsing model."""
        self.logger.info("Loading FashnHumanParser")

        hp_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.hp_model = FashnHumanParser(device=hp_device)

        self.logger.info("FashnHumanParser loaded")

    @torch.inference_mode()
    def _sample(
        self,
        *,
        ca_images: torch.Tensor,
        garment_images: torch.Tensor,
        person_poses: torch.Tensor,
        garment_poses: torch.Tensor,
        garment_categories: torch.Tensor,
        num_timesteps: int = 30,
        time_shift_mu: float = 1.5,
        guidance_scale: float = 1.5,
        garment_guidance_scale: Optional[float] = None,
        person_guidance_scale: Optional[float] = None,
        skip_cfg_last_n_steps: int = 1,
        use_tqdm: bool = True,
    ) -> List[Image.Image]:
        """
        Euler sampling with CFG.

        By default uses the trained/validated joint single-scale CFG. Passing
        `garment_guidance_scale` and/or `person_guidance_scale` switches to an
        EXPERIMENTAL decoupled 3-term CFG (see `TryOnModel.forward_for_decoupled_cfg`)
        that costs ~50% more compute per step (3x batched forward vs. 2x).
        """
        device, dtype = ca_images.device, ca_images.dtype
        batch_size = ca_images.shape[0]

        decoupled = garment_guidance_scale is not None or person_guidance_scale is not None
        garment_scale = garment_guidance_scale if garment_guidance_scale is not None else guidance_scale
        person_scale = person_guidance_scale if person_guidance_scale is not None else guidance_scale

        # Init noisy images
        c, h, w = self.tryon_model.channels_in, *self.tryon_model.input_shape
        images = torch.randn((batch_size, c, h, w), dtype=dtype, device=device)

        # Time schedule (from 0 -> 1)
        timesteps = get_rf_schedule(num_steps=num_timesteps, mu=time_shift_mu)

        model_kwargs = {
            "person_poses": person_poses,
            "garment_poses": garment_poses,
            "ca_images": ca_images,
            "garment_images": garment_images,
            "garment_categories": garment_categories,
        }

        # Euler sampling loop
        for step_idx, (t_curr, t_prev) in enumerate(
            tqdm(
                zip(timesteps[:-1], timesteps[1:]),
                desc="Sampling",
                total=len(timesteps) - 1,
                disable=not use_tqdm,
            )
        ):
            dt = t_prev - t_curr
            t_vec = torch.full((batch_size,), t_curr, dtype=dtype, device=device)
            skip_cfg = skip_cfg_last_n_steps > 0 and step_idx >= num_timesteps - skip_cfg_last_n_steps

            if decoupled:
                pred = self.tryon_model.forward_for_decoupled_cfg(images, t_vec, **model_kwargs)
                v_null, v_garment, v_full = pred["v_null"], pred["v_garment"], pred["v_full"]
                if skip_cfg:
                    v_guided = v_full
                else:
                    v_guided = v_null + garment_scale * (v_garment - v_null) + person_scale * (v_full - v_garment)
            else:
                pred = self.tryon_model.forward_for_cfg(images, t_vec, **model_kwargs)
                v_c, v_u = pred["v_c"], pred["v_u"]
                # Skip CFG at final steps to prevent color saturation
                if skip_cfg:
                    v_guided = v_c
                else:
                    v_guided = v_u + guidance_scale * (v_c - v_u)

            images = images + dt * v_guided

        images = images.to(dtype=torch.float).clamp_(-1.0, 1.0)
        return [tensor_to_pil(img, unnormalize=True) for img in images]

    @torch.inference_mode()
    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        category: Literal["tops", "bottoms", "one-pieces"],
        garment_photo_type: Literal["model", "flat-lay"] = "model",
        num_samples: int = 1,
        num_timesteps: int = 30,
        guidance_scale: float = 1.5,
        garment_guidance_scale: Optional[float] = None,
        person_guidance_scale: Optional[float] = None,
        skip_cfg_last_n_steps: int = 1,
        seed: int = 42,
        segmentation_free: bool = True,
        validate_inputs: bool = True,
        strict_validation: bool = False,
        check_category_match: bool = True,
        strict_category_check: bool = False,
        preserve_background: bool = True,
        restore_resolution: bool = True,
        sharpen: bool = True,
        restore_faces: bool = False,
        harmonize_lighting: bool = True,
    ) -> PipelineOutput:
        """
        Run virtual try-on inference.

        Args:
            person_image: RGB image of the person to dress.
            garment_image: RGB image of the garment (model photo or flat-lay).
            category: Garment category - "tops", "bottoms", or "one-pieces".
            garment_photo_type: "model" if garment is worn by a person,
                "flat-lay" for product shots on plain backgrounds.
            num_samples: Number of output images to generate (1-4). When > 1,
                outputs are ranked by garment-fidelity score (best first).
            num_timesteps: Diffusion sampling steps. Higher = better quality, slower.
                Recommended: 20 (fast), 30 (balanced), 50 (quality).
            guidance_scale: Classifier-free guidance strength.
            garment_guidance_scale: EXPERIMENTAL. If set (with or without
                person_guidance_scale), switches to a decoupled 3-term CFG that
                controls garment fidelity (logos/colors/shape) independently from
                person/pose fidelity, at ~50% extra compute per step. The model was
                trained with joint conditional dropout only — independently dropping
                garment vs. person conditioning was never seen during training, so
                this is an inference-time extrapolation. Validate on real examples
                before relying on it; omit both scales for the trained/validated
                joint-CFG default.
            person_guidance_scale: EXPERIMENTAL, see garment_guidance_scale. Falls
                back to guidance_scale if unset while garment_guidance_scale is set.
            skip_cfg_last_n_steps: Skip CFG for final N steps to prevent color saturation.
            seed: Random seed for reproducibility.
            segmentation_free: If True, generate without masking the person image.
                Recommended for better body preservation and unconstrained garment volume
                (allows garments to expand beyond the original outfit's boundaries).
            validate_inputs: If True, log warnings for low-resolution images, low pose
                confidence, or a person photo that doesn't show the body region the
                requested category needs.
            strict_validation: If True, raise instead of warn on severe input issues
                (e.g. no person detected).
            check_category_match: If True, warn when the garment image's own
                segmentation suggests a different category than the one requested.
            strict_category_check: If True, raise instead of warn on a category mismatch.
            preserve_background: If True, alpha-composite the generated garment region
                back over the pixel-exact original photo, guaranteeing the background,
                face, and other untouched regions never drift from the input. Also
                restores the original input resolution as a side effect.
            restore_resolution: If True (and preserve_background is False), resize
                output back to the original input resolution instead of returning it
                capped at the model's internal working resolution.
            sharpen: If True, apply a mild unsharp mask to the generated content to
                counter typical diffusion-model softness.
            restore_faces: If True, run GFPGAN face restoration on the generated
                content (requires `pip install fashn-vton[enhance]`; no-ops otherwise).
                Off by default: it can subtly alter facial features, and is redundant
                when preserve_background=True since the face is already restored to
                the exact original pixels during compositing. Mainly useful with
                preserve_background=False.
            harmonize_lighting: If True, nudge the generated garment region's
                lightness toward the surrounding context (cheap, no new dependency)
                so a garment shot under different studio lighting than the person
                photo looks less "pasted on". Only lightness is adjusted, not color,
                to avoid degrading garment-color fidelity.

        Returns:
            PipelineOutput with `images` list containing generated PIL Images,
            `scores` (garment-fidelity ranking) when num_samples > 1, `warnings`
            (any input/category issues found), and `edit_mask` (the garment-affected
            region, at the output image's resolution).
        """
        # Set seed
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Pre-resize for pose detection quality
        original_person_image = person_image.convert("RGB")
        person_image = self.pre_resize(person_image, allow_upsampling=False)
        garment_image = self.pre_resize(garment_image, allow_upsampling=False)

        person_image_np = np.array(person_image)
        garment_image_np = np.array(garment_image)

        # Pose detection (DWPose expects BGR)
        person_pose = self.pose_model(person_image_np[..., ::-1])
        garment_pose = (
            get_dummy_dw_keypoints()
            if garment_photo_type == "flat-lay"
            else self.pose_model(garment_image_np[..., ::-1])
        )

        person_pose_img = draw_pose(person_pose, person_image_np.shape[0], person_image_np.shape[1], grayscale=True)
        garment_pose_img = draw_pose(garment_pose, garment_image_np.shape[0], garment_image_np.shape[1], grayscale=True)

        # Human parsing
        person_seg_pred = self.hp_model.predict(person_image_np)
        garment_seg_pred = self.hp_model.predict(garment_image_np)

        warnings: List[str] = []
        if validate_inputs:
            warnings.extend(
                self._validate_inputs(
                    person_image_np, person_pose, person_seg_pred, category, strict=strict_validation
                )
            )
        if check_category_match:
            category_warning = self._check_category_match(garment_seg_pred, category, strict=strict_category_check)
            if category_warning:
                warnings.append(category_warning)

        # Get labels to segment based on category
        body_coverage = CATEGORY_TO_BODY_COVERAGE.get(category)
        labels_to_segment = BODY_COVERAGE_TO_FASHN_LABELS.get(body_coverage)
        labels_to_segment_indices = [FASHN_LABELS_TO_IDS[label] for label in labels_to_segment]

        # Edit-region mask, computed regardless of `segmentation_free` — used later to
        # composite the generated output back over the pixel-exact original image.
        edit_mask = compute_clothing_agnostic_mask(
            seg_pred=person_seg_pred.copy(),
            labels_to_segment_indices=labels_to_segment_indices.copy(),
            body_coverage=body_coverage,
            logger=self.logger,
        )

        # Create clothing-agnostic and garment images
        ca_image = create_clothing_agnostic_image(
            img_np=person_image_np.copy(),
            seg_pred=person_seg_pred.copy(),
            labels_to_segment_indices=labels_to_segment_indices.copy(),
            body_coverage=body_coverage,
            disable_masking=segmentation_free,
            logger=self.logger,
        )

        garment_image_processed = create_garment_image(
            img_np=garment_image_np,
            seg_pred=garment_seg_pred,
            labels_to_segment_indices=labels_to_segment_indices.copy(),
            disable_masking=garment_photo_type == "flat-lay",
        )

        # Resize/pad for model input
        ca_image = self.resize_pad_fn(ca_image, mem_padding=True)
        garment_image_processed = self.resize_pad_fn(garment_image_processed)
        person_pose_img = self.resize_pad_fn(person_pose_img, interpolation=cv2.INTER_NEAREST_EXACT)
        garment_pose_img = self.resize_pad_fn(garment_pose_img, interpolation=cv2.INTER_NEAREST_EXACT)

        # Prepare tensors
        def prepare_tensor(img: np.ndarray) -> torch.Tensor:
            t = numpy_to_torch(img).unsqueeze(0)
            t = normalize_uint8_to_neg1_1(t)
            t = t.to(self.device).repeat(num_samples, 1, 1, 1)
            return t

        ca_tensor = prepare_tensor(ca_image)
        garment_tensor = prepare_tensor(garment_image_processed)
        person_pose_tensor = prepare_tensor(person_pose_img)
        garment_pose_tensor = prepare_tensor(garment_pose_img)

        garment_categories = (
            torch.tensor(self.CATEGORY_TO_LABEL[category]).unsqueeze(0).repeat(num_samples).to(self.device)
        )

        # Cast to inference dtype
        ca_tensor = ca_tensor.to(dtype=self.inference_dtype)
        garment_tensor = garment_tensor.to(dtype=self.inference_dtype)
        person_pose_tensor = person_pose_tensor.to(dtype=self.inference_dtype)
        garment_pose_tensor = garment_pose_tensor.to(dtype=self.inference_dtype)

        # Run sampling
        self.logger.info(f"Running inference with {num_timesteps} timesteps...")
        images = self._sample(
            ca_images=ca_tensor,
            garment_images=garment_tensor,
            person_poses=person_pose_tensor,
            garment_poses=garment_pose_tensor,
            garment_categories=garment_categories,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            garment_guidance_scale=garment_guidance_scale,
            person_guidance_scale=person_guidance_scale,
            skip_cfg_last_n_steps=skip_cfg_last_n_steps,
        )

        # Unpad outputs
        images = [self.resize_pad_fn.unpad(img) for img in images]

        # Optional GFPGAN face restoration (no-ops if the [enhance] extra isn't installed)
        if restore_faces:
            images = [postprocess.restore_face(img) for img in images]

        # Restore original resolution and/or protect background+identity pixels.
        # Compositing resizes to `original_person_image`'s resolution as a side effect.
        if preserve_background:
            images = [
                postprocess.composite_preserve_background(img, original_person_image, edit_mask)
                for img in images
            ]
        elif restore_resolution:
            images = [postprocess.resize_to_match(img, original_person_image.size) for img in images]

        # Nudge the garment region's lighting toward its surroundings (color untouched)
        if harmonize_lighting:
            images = [postprocess.harmonize_lighting(img, edit_mask) for img in images]

        # Sharpen only the model-generated content; re-composite so any protected
        # background pixels stay exactly as they were even after sharpening.
        if sharpen:
            sharpened = [postprocess.unsharp_mask(img) for img in images]
            if preserve_background:
                images = [
                    postprocess.composite_preserve_background(s, base, edit_mask)
                    for s, base in zip(sharpened, images)
                ]
            else:
                images = sharpened

        # Rank multiple samples by garment-fidelity, blended with face-identity
        # similarity when available (best first). Identity scoring only helps
        # distinguish samples when preserve_background=False — with the default
        # preserve_background=True the face is already identical across samples.
        scores = None
        if num_samples > 1:
            garment_scores = [
                postprocess.score_garment_fidelity(img, garment_image_processed, edit_mask) for img in images
            ]
            identity_scores = [postprocess.face_identity_score(img, original_person_image) for img in images]
            scores = [
                g if idn is None else 0.5 * g + 0.5 * idn for g, idn in zip(garment_scores, identity_scores)
            ]
            order = sorted(range(len(images)), key=lambda i: scores[i], reverse=True)
            images = [images[i] for i in order]
            scores = [scores[i] for i in order]

        # Resize edit_mask to the final output resolution so callers (e.g.
        # run_outfit) can use it directly against the returned images.
        output_size = images[0].size if images else original_person_image.size
        edit_mask_out = cv2.resize(
            edit_mask.astype(np.uint8), output_size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        self.logger.info(f"Generated {len(images)} images")

        return PipelineOutput(images=images, scores=scores, warnings=warnings, edit_mask=edit_mask_out)

    def run_outfit(
        self,
        person_image: Image.Image,
        garments: List[Tuple[Image.Image, Literal["tops", "bottoms", "one-pieces"], Literal["model", "flat-lay"]]],
        **kwargs,
    ) -> PipelineOutput:
        """
        Apply multiple garments sequentially onto the same person (e.g. a top, then
        a bottom, for a full outfit in one logical call).

        Each pass regenerates the *entire* canvas from noise — it's not inpainting —
        so without protection a later pass could subtly drift an earlier pass's
        already-correct garment region. This method explicitly re-composites each
        earlier pass's edit region back in after every subsequent pass, on top of
        whatever `preserve_background` already does for the background/identity.
        In practice the per-category masks (e.g. "upper" excludes legs, "lower"
        excludes arms) are already mostly disjoint, so this is a safety net more
        than something that fires constantly.

        Args:
            person_image: RGB image of the person to dress.
            garments: (garment_image, category, garment_photo_type) tuples, applied
                in order. E.g. [(top_img, "tops", "model"), (bottom_img, "bottoms", "flat-lay")].
            **kwargs: Forwarded to every underlying `__call__` (num_samples is
                forced to 1 per pass — ranking/sample-selection isn't meaningful
                mid-sequence; preserve_background defaults to True if unset).

        Returns:
            PipelineOutput for the final composed image: `scores` are from the last
            pass, `warnings` are concatenated across all passes, and `edit_mask`
            covers the union of every pass's edit region.
        """
        if not garments:
            raise ValueError("garments must contain at least one (garment_image, category, garment_photo_type) tuple")

        kwargs = dict(kwargs)
        kwargs["num_samples"] = 1
        kwargs.setdefault("preserve_background", True)

        current_image = person_image.convert("RGB")
        protected_mask: Optional[np.ndarray] = None
        all_warnings: List[str] = []
        last_scores = None

        for garment_image, category, garment_photo_type in garments:
            result = self(
                person_image=current_image,
                garment_image=garment_image,
                category=category,
                garment_photo_type=garment_photo_type,
                **kwargs,
            )
            new_image = result.images[0]
            all_warnings.extend(result.warnings)
            last_scores = result.scores

            if protected_mask is not None:
                # Force the already-finalized region back from the previous pass;
                # keep this pass's fresh edit elsewhere.
                new_image = postprocess.composite_preserve_background(
                    generated=current_image, original=new_image, edit_mask=protected_mask
                )

            current_image = new_image
            protected_mask = result.edit_mask if protected_mask is None else (protected_mask | result.edit_mask)

        return PipelineOutput(
            images=[current_image], scores=last_scores, warnings=all_warnings, edit_mask=protected_mask
        )
