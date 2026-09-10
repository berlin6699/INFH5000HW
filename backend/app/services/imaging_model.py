"""Lightweight, local Chest X-ray inference for the course prototype.

The model is intentionally loaded lazily: the dashboard and all mock agents
remain usable if the optional PyTorch environment or weight file is missing.
Scores are research-model signals, not diagnoses or calibrated probabilities.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from app.schemas import Abnormality, Acuity, ImagingFinding, ImagingLabel, ImagingSourceMode

MODEL_WEIGHTS = "densenet121-res224-all"
MODEL_NAME = "TorchXRayVision DenseNet121 (all datasets, 224px)"
WEIGHT_FILENAME = (
    "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-"
    "d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
)
WEIGHT_PATH = Path.home() / ".torchxrayvision" / "models_data" / WEIGHT_FILENAME

LABEL_MAP: dict[str, ImagingLabel] = {
    "Atelectasis": ImagingLabel.ATELECTASIS,
    "Consolidation": ImagingLabel.CONSOLIDATION,
    "Infiltration": ImagingLabel.FOCAL_OPACITY,
    "Pneumothorax": ImagingLabel.PNEUMOTHORAX,
    "Edema": ImagingLabel.PULMONARY_EDEMA,
    "Emphysema": ImagingLabel.HYPERINFLATION,
    "Fibrosis": ImagingLabel.FIBROSIS,
    "Effusion": ImagingLabel.PLEURAL_EFFUSION,
    "Cardiomegaly": ImagingLabel.CARDIOMEGALY,
    "Nodule": ImagingLabel.NODULE,
    "Mass": ImagingLabel.MASS,
    "Lung Lesion": ImagingLabel.FOCAL_OPACITY,
    "Lung Opacity": ImagingLabel.FOCAL_OPACITY,
}


class ImagingModelError(RuntimeError):
    """Raised when the optional model cannot load or an image is invalid."""


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        import torchxrayvision as xrv
    except (ImportError, OSError) as exc:  # includes Windows DLL load failures
        raise ImagingModelError(
            "本地影像模型依赖未正确安装，请重新运行 start.ps1。"
        ) from exc

    try:
        model = xrv.models.DenseNet(weights=MODEL_WEIGHTS)
        model.eval()
    except Exception as exc:  # download/cache errors vary by platform
        raise ImagingModelError("本地胸片模型或权重加载失败。") from exc
    return model, torch, (xrv, np)


def model_status() -> dict[str, object]:
    """Return a cheap status check without loading the model into memory."""
    try:
        import importlib.util

        dependencies_installed = all(
            importlib.util.find_spec(name) is not None
            for name in ("torch", "torchxrayvision", "PIL")
        )
    except (ImportError, ValueError):
        dependencies_installed = False
    return {
        "available": dependencies_installed and WEIGHT_PATH.exists(),
        "dependencies_installed": dependencies_installed,
        "weights_downloaded": WEIGHT_PATH.exists(),
        "weights_bytes": WEIGHT_PATH.stat().st_size if WEIGHT_PATH.exists() else 0,
        "model": MODEL_NAME,
        "device": "cpu",
        "loaded": _load_model.cache_info().currsize > 0,
    }


def ensure_model_ready() -> None:
    """Install-time warm-up used to fetch weights before the server starts."""
    _load_model()


def analyse_image_bytes(data: bytes, *, image_id: str, study_date) -> ImagingFinding:
    """Run one PNG/JPEG through the model and return the shared agent contract."""
    if not data:
        raise ImagingModelError("上传的图像为空。")

    try:
        from PIL import Image, UnidentifiedImageError

        with Image.open(BytesIO(data)) as source:
            source.load()
            if source.width < 128 or source.height < 128:
                raise ImagingModelError("图像尺寸过小，宽高至少需要 128 像素。")
            gray = source.convert("L")
            pixels = gray.copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImagingModelError("无法读取图像，请上传有效的 PNG 或 JPEG 胸片。") from exc

    model, torch, libs = _load_model()
    xrv, np = libs
    image = np.asarray(pixels, dtype=np.float32)
    image = xrv.datasets.normalize(image, 255)
    image = image[None, :, :]
    image = xrv.datasets.XRayCenterCrop()(image)
    image = xrv.datasets.XRayResizer(224)(image)
    tensor = torch.from_numpy(image).unsqueeze(0)

    try:
        with torch.inference_mode():
            raw_scores = model(tensor)[0].detach().cpu().numpy()
    except Exception as exc:
        raise ImagingModelError("本地胸片模型推理失败。") from exc

    scored = sorted(
        (
            (label, float(score))
            for label, score in zip(model.pathologies, raw_scores, strict=True)
            if np.isfinite(score)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    confidence = {label: round(score, 4) for label, score in scored}
    findings = [f"{label}: model score {score:.1%}" for label, score in scored[:5]]

    # 0.5 is the model package's operating-point-normalised decision boundary.
    # We only map observation labels supported by the application's closed
    # vocabulary, and deliberately leave acuity indeterminate.
    by_label: dict[ImagingLabel, tuple[str, float]] = {}
    for raw_label, score in scored:
        mapped = LABEL_MAP.get(raw_label)
        if mapped is None or score < 0.5:
            continue
        current = by_label.get(mapped)
        if current is None or score > current[1]:
            by_label[mapped] = (raw_label, score)

    abnormalities = [
        Abnormality(
            label=label,
            description=f"Research-model signal: {raw_label} (score {score:.1%})",
            acuity=Acuity.INDETERMINATE,
            severity_score=min(5, max(1, round(score * 5))),
            confidence=score,
        )
        for label, (raw_label, score) in sorted(
            by_label.items(), key=lambda item: item[1][1], reverse=True
        )
    ]
    summary = (
        f"本地研究模型有 {len(abnormalities)} 个映射征象达到显示阈值；"
        "分数仅供课程演示，不能视为诊断或临床概率。"
    )
    return ImagingFinding(
        image_id=image_id,
        study_date=study_date,
        source_mode=ImagingSourceMode.REAL_MODEL,
        provenance=(
            f"由本机 CPU 上的 {MODEL_NAME} 对用户上传图像生成；"
            "未调用外部影像 API。"
        ),
        findings=findings,
        abnormalities=abnormalities,
        confidence=confidence,
        has_acute_abnormality=False,
        summary=summary,
    )
