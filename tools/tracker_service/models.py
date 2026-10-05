"""Model runtime for the standalone synoptics tracker service.

Two models, one job each:

* **Grounder** — open-vocabulary acquisition/re-acquisition from the free-text target. Selected once at
  import by ``TRACKER_GROUNDER``: ``gdino`` (default, GroundingDINO Swin-T) or ``sam3`` (SAM 3 image model,
  text prompt -> instance boxes, run under BF16 autocast). Anything else is a startup error.
* **SAM 2.1** ``hiera_base_plus`` — the per-frame tracker, driven by the causal adapter proven by the P0
  spike (exactly one frame ingested per step, memory retained, no future frame materialised, bounded
  physical state). The same for both grounders.

Everything here is pinned: the SAM 2 source revision, the checkpoints, and the library override list.
The checkpoints live on the Docker-partition volume (``/v``); nothing is downloaded at request time.

Deliberate configuration choices (do not "improve" silently):

* ``model.fill_hole_area = 0`` is set **explicitly** through the pinned builder's own override list, ordered
  last so it genuinely wins, and the resolved config **and** the runtime attribute are asserted. Without the
  compiled ``sam2._C`` extension the library's default (``fill_hole_area=8``) silently skips hole filling
  every frame; 12-frame parity measured 0 differing pixels between the two paths, so this is an intentional,
  verified choice rather than an implicit failure path.
* GroundingDINO thresholds are the pinned library's documented example values, declared here as constants and
  never tuned against any reference. ``TRACKER_BOX_THRESHOLD`` overrides the box threshold for either
  grounder (a measured 2026-10-04 bench put the useful GroundingDINO range at 0.4-0.45; the default stays 0.3).
* ``TRACKER_SAM2_COMPILE=image_encoder`` wraps the SAM 2 image encoder in ``torch.compile`` (mode ``default``,
  ``fullgraph``, static shapes: the encoder input is always ``IMAGE_SIZE`` square). Off by default. Measured
  2026-10-04 on torch 2.14.1+cu132 (5 clips, 10 fps closed loop): tracking step p50 16.8 -> 15.0 ms, boxes
  equal to the eager path except on frames where the mask flips between two shapes astra graded both correct.
  The compile happens on the first encoder call, so the service warms up before it reports ready.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

PYDEPS = os.environ.get("TRACKER_PYDEPS", "/v/tracker/pydeps")
SAM2_SRC = os.environ.get("TRACKER_SAM2_SRC", "/v/tracker/src")
for _p in (PYDEPS, SAM2_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HOME", os.environ.get("TRACKER_HF_HOME", "/v/tracker/hf"))

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CKPT = os.environ.get("TRACKER_SAM2_CKPT", "/v/tracker/models/sam2.1_hiera_base_plus.pt")
SAM2_CKPT_SHA = "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
SAM2_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"

GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GROUNDING_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
GROUNDING_CKPT_SHA = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"

SAM3_MODEL_ID = "facebook/sam3"           # gated on the Hub: the weights must already be in HF_HOME
SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
SAM3_CKPT_SHA = "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a"

GROUNDERS = ("gdino", "sam3")


def _grounder() -> str:
    """Which acquisition model to load. Read once at import and validated fail-closed, like the window."""
    raw = os.environ.get("TRACKER_GROUNDER")
    if raw is None or raw.strip() == "":
        return "gdino"
    value = raw.strip()
    if value not in GROUNDERS:
        raise RuntimeError(f"TRACKER_GROUNDER must be one of {GROUNDERS}, got {raw!r}")
    return value


GROUNDER = _grounder()

# Declared thresholds (library example values; predeclared, never tuned against a reference).
BOX_THRESHOLD = float(os.environ.get("TRACKER_BOX_THRESHOLD", "0.3"))
TEXT_THRESHOLD = float(os.environ.get("TRACKER_TEXT_THRESHOLD", "0.25"))
# Two boxes of the same label whose overlap is below this are treated as DISTINCT instances (ambiguity).
DISTINCT_INSTANCE_IOU = 0.5
# Occlusion/loss from the model's own object-presence head (sigmoid of object_score_logits).
TRACK_KEEP_SCORE = 0.5


def _occluded_max_frames() -> int:
    """Occlusion window, in PROCESSED consecutive non-observable frames (not camera seconds).

    The 16th such frame with the default 15 turns the run ``lost``. Read once at import from
    ``TRACKER_OCCLUDED_MAX_FRAMES`` and validated fail-closed: anything that is not a plain integer in
    ``1..600`` is a startup error, never a silent fallback to the default.
    """
    raw = os.environ.get("TRACKER_OCCLUDED_MAX_FRAMES")
    if raw is None or raw.strip() == "":
        return 15
    text = raw.strip()
    if not text.isascii() or not text.isdigit():
        raise RuntimeError(f"TRACKER_OCCLUDED_MAX_FRAMES must be an integer 1..600, got {raw!r}")
    value = int(text)
    if not 1 <= value <= 600:
        raise RuntimeError(f"TRACKER_OCCLUDED_MAX_FRAMES must be an integer 1..600, got {raw!r}")
    return value


OCCLUDED_MAX_FRAMES = _occluded_max_frames()
# After a real LOSS nothing is re-acquired silently (no threshold can prove the same object): the run
# stays lost until the user re-selects with a new seeded run. Transient occlusion inside the window is
# recovered through SAM's own memory, with no reset.

SAM2_COMPILE_TARGETS = ("off", "image_encoder")


def _sam2_compile() -> str:
    """Which SAM 2 component to ``torch.compile``. Read once at import, validated fail-closed."""
    raw = os.environ.get("TRACKER_SAM2_COMPILE")
    if raw is None or raw.strip() == "":
        return "off"
    value = raw.strip()
    if value not in SAM2_COMPILE_TARGETS:
        raise RuntimeError(f"TRACKER_SAM2_COMPILE must be one of {SAM2_COMPILE_TARGETS}, got {raw!r}")
    return value


SAM2_COMPILE = _sam2_compile()

IMAGE_SIZE = 1024                     # pinned config: model.image_size
KEEP_NONCOND = 16                     # == model.max_obj_ptrs_in_encoder (object-pointer window)
KEEP_FRAMES = 3                       # retained frame tensors (only the current one is ever read)

POLICY = {
    "box_threshold": BOX_THRESHOLD,
    "text_threshold": TEXT_THRESHOLD,
    "distinct_instance_iou": DISTINCT_INSTANCE_IOU,
    "track_keep_score": TRACK_KEEP_SCORE,
    "occluded_max_frames": OCCLUDED_MAX_FRAMES,
    "occluded_max_frames_source": "env" if os.environ.get("TRACKER_OCCLUDED_MAX_FRAMES", "").strip() else "default",
    "lost_policy": "no silent re-acquire: stays lost until a user-seeded run",
    "keep_noncond": KEEP_NONCOND,
    "keep_frames": KEEP_FRAMES,
    "sam2_revision": SAM2_REVISION,
    "sam2_compile": SAM2_COMPILE,
    "grounder": GROUNDER,
    "grounding_model": SAM3_MODEL_ID if GROUNDER == "sam3" else GROUNDING_MODEL_ID,
    "grounding_revision": SAM3_REVISION if GROUNDER == "sam3" else GROUNDING_REVISION,
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- SAM 2.1 (tracker)
def _postprocess_available() -> bool:
    try:
        from sam2 import _C  # noqa: F401
        return True
    except Exception:
        return False


def verify_sam2_checkpoint() -> dict[str, Any]:
    """Explicit, fail-closed checkpoint pin: a missing or altered file is a startup error, not a warning."""
    if not os.path.exists(SAM2_CKPT):
        raise RuntimeError(f"SAM 2.1 checkpoint missing: {SAM2_CKPT}")
    got = _sha256(SAM2_CKPT)
    if got != SAM2_CKPT_SHA:
        raise RuntimeError(f"SAM 2.1 checkpoint digest mismatch: got {got}, want {SAM2_CKPT_SHA}")
    return {"path": SAM2_CKPT, "sha256": got, "expected": SAM2_CKPT_SHA, "ok": True}


def verify_grounding_checkpoint() -> dict[str, Any]:
    """Same pin for the grounder: resolved through the pinned revision and digested before use."""
    from huggingface_hub import hf_hub_download

    repo, revision, want = (
        (SAM3_MODEL_ID, SAM3_REVISION, SAM3_CKPT_SHA) if GROUNDER == "sam3"
        else (GROUNDING_MODEL_ID, GROUNDING_REVISION, GROUNDING_CKPT_SHA)
    )
    path = hf_hub_download(repo_id=repo, filename="model.safetensors", revision=revision)
    got = _sha256(path)
    if got != want:
        raise RuntimeError(f"{repo} checkpoint digest mismatch: got {got}, want {want}")
    return {"path": path, "sha256": got, "expected": want, "size": os.path.getsize(path), "ok": True}


def load_sam2(device: str = "cuda"):
    """Build the pinned SAM 2.1 video predictor with an explicit, verified ``fill_hole_area=0``."""
    import torch
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    os.chdir(SAM2_SRC)
    from sam2.build_sam import _load_checkpoint

    pin = verify_sam2_checkpoint()
    overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        "++model.binarize_mask_from_pts_for_mem_enc=true",
        "++model.fill_hole_area=8",       # library default, kept so the ordering stays explicit
        "model.fill_hole_area=0",         # ours, LAST so it wins
    ]
    cfg = compose(config_name=SAM2_CONFIG, overrides=overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, SAM2_CKPT)
    model = model.to(device)
    model.eval()
    if int(cfg.model.fill_hole_area) != 0 or int(model.fill_hole_area) != 0:
        raise RuntimeError("fill_hole_area override did not take effect")
    if SAM2_COMPILE == "image_encoder":
        model.image_encoder.forward = torch.compile(
            model.image_encoder.forward, mode="default", fullgraph=True, dynamic=False
        )
    model._tracker_meta = {  # type: ignore[attr-defined]
        "fill_hole_area": int(model.fill_hole_area),
        "image_size": int(model.image_size),
        "postprocess_extension_available": _postprocess_available(),
        "checkpoint": pin,
        "device": device,
        "compile": SAM2_COMPILE,
        "torch": torch.__version__,
    }
    return model


class FrameStore:
    """Adapter-owned frame store: logical length == frames ingested; physical contents bounded."""

    def __init__(self, bounded: int | None = None) -> None:
        self._frames: dict[int, Any] = {}
        self.ingested = -1
        self.bounded = bounded

    def append(self, idx: int, tensor: Any) -> None:
        if idx != self.ingested + 1:
            raise AssertionError(f"non-sequential ingest {idx} after {self.ingested}")
        self._frames[idx] = tensor
        self.ingested = idx
        self.prune_physical()

    def prune_physical(self) -> None:
        if not self.bounded:
            return
        floor = self.ingested - self.bounded + 1
        for j in [j for j in self._frames if j < floor]:
            del self._frames[j]

    def __getitem__(self, idx: int) -> Any:
        if idx > self.ingested:
            raise LookupError(f"frame {idx} not ingested yet (ingested up to {self.ingested})")
        if idx not in self._frames:
            raise KeyError(f"frame {idx} was pruned from the bounded store")
        return self._frames[idx]

    def __len__(self) -> int:
        return self.ingested + 1

    @property
    def physical(self) -> int:
        return len(self._frames)


def _make_state(store: FrameStore, video_h: int, video_w: int, device: Any) -> dict[str, Any]:
    return {
        "images": store,
        "num_frames": 0,
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
        "video_height": video_h,
        "video_width": video_w,
        "device": device,
        "storage_device": device,
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
    }


def frame_to_tensor(jpg_bytes: bytes):
    """Decode a JPEG to a native-resolution ``uint8`` HWC tensor — the resize, the cast and the
    ``/255.0`` scaling all happen later on the GPU (see ``CausalSam2Session._prepare``).

    Only the decoder's own uint8 output crosses to the device (1.31 MB for the 810x540 fixture) instead of
    a CPU-resized float32 image (12.58 MB).

    Geometry is preserved; the resampling is not. The CPU path used ``PIL.Image.resize`` (Pillow's own
    filter) and the GPU path uses bilinear interpolation with anti-aliasing, so the frame that reaches the
    model is a **re-resampled** version of the same image, not bit-identical to it. What is unchanged is the
    order and the geometry: resize to ``image_size``, then the same standardisation.
    """
    import numpy as np
    import torch
    from PIL import Image

    img_pil = Image.open(io.BytesIO(jpg_bytes))
    video_width, video_height = img_pil.size
    # ``np.array`` (not ``asarray``): the PIL buffer is read-only, and ``torch.from_numpy`` warns on a
    # non-writable array. This makes a writable, owned copy instead of borrowing PIL's buffer.
    arr = np.array(img_pil.convert("RGB"))
    if arr.dtype != np.uint8:
        raise ValueError(f"unexpected frame dtype {arr.dtype}")
    return torch.from_numpy(arr), video_height, video_width


@dataclass(slots=True)
class StepResult:
    box_norm_xywh: dict[str, float] | None
    score: float                    # sigmoid(object_score_logits) — uncalibrated presence score
    area_px: int
    infer_ms: float


class CausalSam2Session:
    """One track: exactly one frame ingested per step, memory retained, physical state bounded.

    Proven by the P0 spike on the 289-frame clip: 289/289 single-inference steps, 0 ``reset_state`` calls,
    per-frame mask IoU 1.0 against an unpruned causal reference at the same ingested ``num_frames``, and a
    physical plateau of 3 frame tensors / 16 non-cond outputs / 16 ``frames_tracked`` / 1 conditioning frame.
    """

    def __init__(self, predictor: Any, obj_id: int = 0) -> None:
        self.predictor = predictor
        self.obj_id = obj_id
        self.store = FrameStore(bounded=KEEP_FRAMES)
        self.state: dict[str, Any] | None = None
        self.mean = None
        self.std = None

    # -- internals ------------------------------------------------
    def _prepare(self, jpg_bytes: bytes):
        import torch

        t_u8, video_h, video_w = frame_to_tensor(jpg_bytes)
        device = next(self.predictor.parameters()).device
        if self.mean is None:
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(3, 1, 1)
        # One host-to-device copy of the decoder's uint8 frame; the resize, the float cast, the /255.0
        # scaling and the standardisation all run on the GPU. The state still records the frame's ORIGINAL
        # height/width, so SAM returns its masks at native resolution and the normalised box is unchanged.
        t_f = t_u8.to(device).permute(2, 0, 1).to(torch.float32).div_(255.0).unsqueeze(0)
        img = torch.nn.functional.interpolate(
            t_f, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False, antialias=True
        ).squeeze(0)
        img.sub_(self.mean).div_(self.std)
        if self.state is None:
            self.state = _make_state(self.store, video_h, video_w, device)
        elif (self.state["video_height"], self.state["video_width"]) != (video_h, video_w):
            raise ValueError("frame size changed mid-run; start a new run")
        idx = len(self.store)
        self.store.append(idx, img)
        self.state["num_frames"] = idx + 1
        return idx

    def _prune(self, k: int) -> None:
        floor = k - KEEP_NONCOND
        state = self.state
        assert state is not None
        for od in state["output_dict_per_obj"].values():
            for j in [j for j in od["non_cond_frame_outputs"] if j <= floor]:
                del od["non_cond_frame_outputs"][j]
        for ft in state["frames_tracked_per_obj"].values():
            for j in [j for j in ft if j <= floor]:
                del ft[j]
        for t in [t for t in state["cached_features"] if t <= k - KEEP_FRAMES]:
            del state["cached_features"][t]
        self.store.prune_physical()

    def _consume(self, k: int, started: float) -> tuple[StepResult, float]:
        device = next(self.predictor.parameters()).device
        gen = self.predictor.propagate_in_video(self.state, start_frame_idx=k, max_frame_num_to_track=0)
        fidx, _obj_ids, masks = next(gen)
        if fidx != k:
            raise RuntimeError(f"tracker produced frame {fidx}, expected {k}")
        for _extra in gen:
            raise RuntimeError("tracker produced more than one frame in a single step")
        # The reduction's single host copy blocks until the outputs it reads are complete, so the work whose
        # result is returned has finished by the time it returns. No device-wide barrier here: it would wait
        # on unrelated streams and inflate the measured step without changing any returned value.
        box, score, area = reduce_mask_single_sync(
            masks, presence_logits(self.state, self.obj_id, k), device
        )
        ms = (time.perf_counter() - started) * 1e3
        self._prune(k)
        return StepResult(box, score, area, ms), ms

    # -- public ---------------------------------------------------
    def acquire(self, jpg_bytes: bytes, box_norm_xywh: dict[str, float]) -> StepResult:
        """Cold start / re-acquire from a normalised box: converted to PIXELS exactly once."""
        started = time.perf_counter()
        k = self._prepare(jpg_bytes)
        state = self.state
        assert state is not None
        if k != 0:
            raise RuntimeError("acquire is only valid on the first frame of a session")
        w = state["video_width"]
        h = state["video_height"]
        x0 = box_norm_xywh["x"] * w
        y0 = box_norm_xywh["y"] * h
        x1 = (box_norm_xywh["x"] + box_norm_xywh["width"]) * w
        y1 = (box_norm_xywh["y"] + box_norm_xywh["height"]) * h
        self.predictor.add_new_points_or_box(
            inference_state=state, frame_idx=k, obj_id=self.obj_id,
            box=[x0, y0, x1, y1], clear_old_points=True,   # normalize_coords defaults True == pixels
        )
        return self._consume(k, started)[0]

    def step(self, jpg_bytes: bytes) -> StepResult:
        started = time.perf_counter()
        k = self._prepare(jpg_bytes)
        return self._consume(k, started)[0]


def presence_logits(state: dict[str, Any], obj_id: int = 0, frame_idx: int | None = None):
    """``object_score_logits`` for a frame — conditioning frames first. One lookup order, one place."""
    k = frame_idx if frame_idx is not None else state["num_frames"] - 1
    for storage in ("cond_frame_outputs", "non_cond_frame_outputs"):
        out = state["output_dict_per_obj"][obj_id][storage].get(k)
        if out is not None and out.get("object_score_logits") is not None:
            return out["object_score_logits"]
    return None


def reduce_mask_single_sync(masks: Any, pres_logit: Any,
                            device: Any) -> tuple[dict[str, float] | None, float, int]:
    """Box, presence score and area for one frame mask, from exactly ONE device-to-host copy.

    The extremes come from fixed-shape row/col ``any()`` projections rather than ``nonzero``, so nothing is
    allocated at a shape that depends on the mask's content, and the presence score rides along in the same
    packed copy instead of taking a synchronisation of its own. The result contract is unchanged from the
    previous CPU path: an empty mask is ``(None, score, 0)``, the box is rounded to 6 decimals, and a
    zero-extent axis is widened to one pixel.
    """
    import torch

    pres_score_t = (
        torch.sigmoid(pres_logit.float().mean())
        if pres_logit is not None
        else torch.tensor(0.0, device=device)
    )

    bin_gpu = (masks > 0.0).squeeze()
    h, w = bin_gpu.shape

    any_row = bin_gpu.any(dim=1)
    any_col = bin_gpu.any(dim=0)
    has_pixels = any_row.any()

    row_idx = torch.arange(h, device=device, dtype=torch.int32)
    col_idx = torch.arange(w, device=device, dtype=torch.int32)

    valid_rows_min = torch.where(any_row, row_idx, h)
    valid_rows_max = torch.where(any_row, row_idx, -1)
    valid_cols_min = torch.where(any_col, col_idx, w)
    valid_cols_max = torch.where(any_col, col_idx, -1)

    y0 = valid_rows_min.amin()
    y1 = valid_rows_max.amax() + 1
    x0 = valid_cols_min.amin()
    x1 = valid_cols_max.amax() + 1
    area = bin_gpu.sum(dtype=torch.int32)

    pack = torch.stack([
        x0.float(), y0.float(), x1.float(), y1.float(),
        area.float(), pres_score_t.float(), has_pixels.float(),
    ])
    x0_f, y0_f, x1_f, y1_f, area_f, score_f, has_px_f = pack.cpu().tolist()

    if has_px_f < 0.5:
        return None, float(score_f), 0
    return (
        {
            "x": round(x0_f / w, 6),
            "y": round(y0_f / h, 6),
            "width": round(max(x1_f - x0_f, 1) / w, 6),
            "height": round(max(y1_f - y0_f, 1) / h, 6),
        },
        float(score_f),
        int(area_f),
    )


# --------------------------------------------------------------------------- GroundingDINO (acquisition)
@dataclass(slots=True)
class Candidate:
    box_norm_xywh: dict[str, float]
    score: float
    phrase: str


def load_grounding(device: str = "cuda"):
    pin = verify_grounding_checkpoint()
    if GROUNDER == "sam3":
        from transformers import Sam3Model, Sam3Processor

        processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, revision=SAM3_REVISION)
        # FP32 weights, BF16 autocast at call time: the measured configuration (top-1 boxes matched FP32 at
        # median IoU 0.996 over 200 bench frames). Casting the weights to BF16 was not measured.
        model = Sam3Model.from_pretrained(SAM3_MODEL_ID, revision=SAM3_REVISION).to(device).eval()
        model._tracker_meta = {"checkpoint": pin, "grounder": GROUNDER, "box_threshold": BOX_THRESHOLD,
                               "autocast": "bf16"}  # type: ignore[attr-defined]
        return processor, model
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(GROUNDING_MODEL_ID, revision=GROUNDING_REVISION)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_MODEL_ID, revision=GROUNDING_REVISION
    ).to(device).eval()
    model._tracker_meta = {"checkpoint": pin, "grounder": GROUNDER, "box_threshold": BOX_THRESHOLD,
                           "text_threshold": TEXT_THRESHOLD}  # type: ignore[attr-defined]
    return processor, model


def ground(processor: Any, model: Any, image: Any, text: str) -> list[Candidate]:
    """Return candidates for ``text`` in the image, best first, using the declared thresholds."""
    import torch

    w, h = image.size
    if GROUNDER == "sam3":
        inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**inputs)
        res = processor.post_process_object_detection(outputs, threshold=BOX_THRESHOLD, target_sizes=[(h, w)])[0]
        boxes = res["boxes"].float()
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)   # SAM 3 boxes can overshoot the frame edge
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        res = {"boxes": boxes, "scores": res["scores"].float(), "labels": [text] * len(res["scores"])}
    else:
        if not text.endswith((".", "?", "!")):
            text = text + "."          # the model expects a sentence-like prompt
        inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        res = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[(h, w)],
        )[0]
    out: list[Candidate] = []
    for box, score, label in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"]):
        x0, y0, x1, y1 = box
        out.append(Candidate(
            box_norm_xywh={
                "x": round(x0 / w, 6), "y": round(y0 / h, 6),
                "width": round(max(x1 - x0, 1) / w, 6), "height": round(max(y1 - y0, 1) / h, 6),
            },
            score=round(float(score), 4), phrase=str(label),
        ))
    out.sort(key=lambda c: -c.score)
    return out


def box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax0, ay0 = a["x"], a["y"]
    ax1, ay1 = a["x"] + a["width"], a["y"] + a["height"]
    bx0, by0 = b["x"], b["y"]
    bx1, by1 = b["x"] + b["width"], b["y"] + b["height"]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return 0.0 if union <= 0 else inter / union


def distinct_instances(candidates: list[Candidate]) -> bool:
    """True when the top candidates are separate objects, not the same object detected twice."""
    if len(candidates) < 2:
        return False
    best = candidates[0]
    return any(box_iou(best.box_norm_xywh, c.box_norm_xywh) < DISTINCT_INSTANCE_IOU for c in candidates[1:])
