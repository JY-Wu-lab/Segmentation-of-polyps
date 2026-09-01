from __future__ import annotations

import copy
import json
from pathlib import Path
from pprint import pformat


OUTPUT_DIR = Path(__file__).resolve().parent

NOTEBOOK_SPECS = (
    {
        "model_name": "FPN+V3",
        "architecture": "FPN",
        "output_name": "FPN+V3-Raw-vs-FMM-Bad-Case-Analysis.ipynb",
        "reference": {
            "CVC-300": {
                "RR": {"Dice": 0.8774025142192841, "IoU": 0.7818476259708405},
                "FF": {"Dice": 0.8772180080413818, "IoU": 0.7846458554267883},
            },
            "CVC-ClinicDB": {
                "RR": {"Dice": 0.8993376493453979, "IoU": 0.8186468183994293},
                "FF": {"Dice": 0.8752999901771545, "IoU": 0.7811286151409149},
            },
            "CVC-ColonDB": {
                "RR": {"Dice": 0.7392354893187681, "IoU": 0.6256391449520985},
                "FF": {"Dice": 0.7796898459394773, "IoU": 0.6670379005372524},
            },
            "ETIS-LaribPolypDB": {
                "RR": {"Dice": 0.7240193273339953, "IoU": 0.5855646218572345},
                "FF": {"Dice": 0.6833710713045937, "IoU": 0.5379231508289065},
            },
            "Kvasir": {
                "RR": {"Dice": 0.8738004118204117, "IoU": 0.7766684889793396},
                "FF": {"Dice": 0.8836412876844406, "IoU": 0.7926690280437469},
            },
        },
    },
    {
        "model_name": "Linknet+V3",
        "architecture": "Linknet",
        "output_name": "Linknet+V3-Raw-vs-FMM-Bad-Case-Analysis.ipynb",
        "reference": {
            "CVC-300": {
                "RR": {"Dice": 0.8605987429618835, "IoU": 0.7603500485420227},
                "FF": {"Dice": 0.895767867565155, "IoU": 0.8122174739837646},
            },
            "CVC-ClinicDB": {
                "RR": {"Dice": 0.8835867643356323, "IoU": 0.7947163879871368},
                "FF": {"Dice": 0.9049492180347443, "IoU": 0.8278017640113831},
            },
            "CVC-ColonDB": {
                "RR": {"Dice": 0.7571494442721208, "IoU": 0.6407776350776354},
                "FF": {"Dice": 0.773495032141606, "IoU": 0.6696476743867},
            },
            "ETIS-LaribPolypDB": {
                "RR": {"Dice": 0.7130793290478843, "IoU": 0.5729481322424752},
                "FF": {"Dice": 0.6868536812918526, "IoU": 0.5396667080266135},
            },
            "Kvasir": {
                "RR": {"Dice": 0.8723773211240768, "IoU": 0.7757739126682281},
                "FF": {"Dice": 0.8569844961166382, "IoU": 0.7499537616968155},
            },
        },
    },
)


INTRO_TEMPLATE = r'''# __MODEL_NAME__：独立运行的 Raw/FMM 坏案例分析

本 notebook 完全独立，仅进行推理和坏案例分析：不会导入或执行其他 notebook，不读取训练集，也不会重新训练模型。

请在原模型使用的 TensorFlow/Keras 环境中，从上到下依次运行全部单元。如果项目位置发生变化，只需修改配置单元中的 `PROJECT_ROOT`。

四条推理路径定义如下：

- **RR**：Raw训练权重 + Raw输入
- **RF**：Raw训练权重 + FMM输入
- **FR**：FMM训练权重 + Raw输入
- **FF**：FMM训练权重 + FMM输入

`FF - RR` 表示论文中完整Raw/FMM流程的变化；`RF - RR` 表示固定Raw权重时，输入修复本身带来的直接变化。

所有结果统一写入当前 notebook 运行目录下新建的 `bad_case_analysis_results/__MODEL_NAME__/` 文件夹。
'''


CONFIG_TEMPLATE = r'''# ============================================================
# 1. 导入依赖与可修改配置
# ============================================================
import gc
import math
import os
import re
from pathlib import Path

os.environ.setdefault("SM_FRAMEWORK", "tf.keras")

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import spearmanr
from tensorflow.keras import Model
from IPython.display import display

import segmentation_models as sm
sm.set_framework("tf.keras")


MODEL_NAME = "__MODEL_NAME__"
MODEL_ARCHITECTURE = "__ARCHITECTURE__"

# 如果原电脑上的项目位置发生变化，只修改这一项即可。
PROJECT_ROOT = Path("D:/WJY/paper1")

IMAGE_HEIGHT = 288
IMAGE_WIDTH = 384
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
PERCENTILE = 99.5
INPAINT_RADIUS = 10
JPEG_QUALITY = 95
PREDICTION_THRESHOLD = 0.5
BOUNDARY_DILATION_RATIO = 0.02
INFERENCE_BATCH_SIZE = 32
METRIC_BATCH_SIZE = 32
SEED = 42

TOP_K = 5
FIGURE_DPI = 300
REFERENCE_TOLERANCE = 1e-6

TEST_DATASETS = (
    "CVC-300",
    "CVC-ClinicDB",
    "CVC-ColonDB",
    "ETIS-LaribPolypDB",
    "Kvasir",
)

REFERENCE_RESULTS = __REFERENCE_RESULTS__

# Jupyter通常以notebook所在目录作为工作目录，结果文件夹将在此处创建。
NOTEBOOK_DIR = Path.cwd().resolve()
SAFE_MODEL_NAME = re.sub(r"[^A-Za-z0-9._+-]+", "_", MODEL_NAME)
RESULT_DIR = NOTEBOOK_DIR / "bad_case_analysis_results" / SAFE_MODEL_NAME
FIGURE_DIR = RESULT_DIR / "figures"
CASE_FIGURE_DIR = FIGURE_DIR / "cases"
for directory in (RESULT_DIR, FIGURE_DIR, CASE_FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print("Model:", MODEL_NAME)
print("Project root:", PROJECT_ROOT.resolve())
print("Result directory:", RESULT_DIR.resolve())
'''


MODEL_CODE = r'''# ============================================================
# 2. 模型结构与历史汇总指标计算方式
# ============================================================
def build_model():
    """建立原实验使用的InceptionV3编码器分割模型，不下载预训练权重。"""
    builders = {
        "FPN": sm.FPN,
        "Linknet": sm.Linknet,
    }
    if MODEL_ARCHITECTURE not in builders:
        raise ValueError(f"Unsupported architecture: {MODEL_ARCHITECTURE}")
    return builders[MODEL_ARCHITECTURE](
        "inceptionv3",
        classes=1,
        activation="sigmoid",
        encoder_weights=None,
    )


def soft_dice_tensor(y_true, y_pred, smooth=1e-7):
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    denominator = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    return (2.0 * intersection + smooth) / (denominator + smooth)


def aggregate_soft_metrics(y_true, y_pred):
    """复现原代码按batch=32计算后再取均值的Soft Dice/IoU口径。"""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Prediction shape mismatch: {y_pred.shape} != {y_true.shape}")
    dice_batches = []
    iou_batches = []
    for start in range(0, len(y_true), METRIC_BATCH_SIZE):
        stop = min(start + METRIC_BATCH_SIZE, len(y_true))
        true_batch = tf.convert_to_tensor(y_true[start:stop], dtype=tf.float32)
        pred_batch = tf.convert_to_tensor(y_pred[start:stop], dtype=tf.float32)
        dice_batches.append(float(soft_dice_tensor(true_batch, pred_batch).numpy()))
        iou_batches.append(float(sm.metrics.iou_score(true_batch, pred_batch).numpy()))
    return {
        "Dice": float(np.mean(dice_batches)),
        "IoU": float(np.mean(iou_batches)),
    }
'''


DATA_CODE = r'''# ============================================================
# 3. 测试数据读取与严格复现的Raw/FMM预处理
# ============================================================
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TEST_ROOT = PROJECT_ROOT / "Data" / "Test"


def list_image_names(image_dir):
    """保留原代码的os.listdir顺序，避免batch分组顺序变化影响汇总指标。"""
    names = [
        name
        for name in os.listdir(image_dir)
        if (Path(image_dir) / name).is_file()
        and Path(name).suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not names:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return names


def validate_image_mask_pairs(image_dir, mask_dir, names):
    missing = [name for name in names if not (Path(mask_dir) / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{mask_dir} is missing {len(missing)} masks, for example {missing[:5]}"
        )


def read_raw_image(image_path):
    """复现Raw分支：直接缩放到384x288、BGR转RGB、归一化到[0, 1]。"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = cv2.resize(image, IMAGE_SIZE)
    image = image[..., ::-1]  # BGR to RGB
    return image.astype(np.float32) / 255.0


def create_highlight_mask(image_path, percentile=PERCENTILE):
    """按灰度图的99.5%分位值生成二值高光mask。"""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot read grayscale image: {image_path}")
    # 与原notebook一致：阈值T=P_99.5(I_g)，THRESH_BINARY保留严格大于T的像素。
    threshold = np.percentile(gray, percentile)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask


def read_fmm_image(image_path):
    """严格复现：99.5%高光mask + TELEA/FMM修复半径10 + JPEG质量95。"""
    source = cv2.imread(str(image_path))
    if source is None:
        raise ValueError(f"Cannot read image: {image_path}")
    # 步骤1：在原始灰度图上计算99.5%分位阈值并生成高光mask。
    highlight_mask = create_highlight_mask(image_path, percentile=PERCENTILE)

    # 步骤2：原图和高光mask均使用INTER_CUBIC缩小到0.6倍。
    # 重要：mask缩放后不再次二值化，这一点严格保留原notebook行为。
    resized_source = cv2.resize(
        source, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC
    )
    resized_mask = cv2.resize(
        highlight_mask, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC
    )
    # 步骤3：使用OpenCV TELEA实现的Fast Marching Method，修复半径r=10。
    restored = cv2.inpaint(
        resized_source, resized_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA
    )
    # 步骤4：按原代码默认的线性插值恢复到384x288。
    restored = cv2.resize(restored, IMAGE_SIZE)

    # 步骤5：原流程会将修复图保存为JPG后再读取；此处以内存编码复现，
    # 质量固定为95，不生成中间文件。
    encoded_ok, encoded = cv2.imencode(
        ".jpg", restored, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not encoded_ok:
        raise RuntimeError(f"Cannot encode FMM image: {image_path}")
    restored = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    # 步骤6：JPG重新读取后为BGR，转换为RGB并归一化到[0, 1]。
    restored = restored[..., ::-1]
    return restored.astype(np.float32) / 255.0


def validate_fmm_configuration():
    """防止后续修改意外改变论文使用的FMM关键参数。"""
    expected = {
        "PERCENTILE": 99.5,
        "INPAINT_RADIUS": 10,
        "JPEG_QUALITY": 95,
        "IMAGE_SIZE": (384, 288),
    }
    observed = {
        "PERCENTILE": PERCENTILE,
        "INPAINT_RADIUS": INPAINT_RADIUS,
        "JPEG_QUALITY": JPEG_QUALITY,
        "IMAGE_SIZE": IMAGE_SIZE,
    }
    if observed != expected:
        raise AssertionError(
            f"FMM配置与原notebook不一致：observed={observed}, expected={expected}"
        )
    print(
        "FMM配置检查通过：99.5% mask，缩放0.6，"
        "cv2.INPAINT_TELEA，r=10，JPEG=95，输出384x288"
    )


validate_fmm_configuration()


def read_ground_truth(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    mask = cv2.resize(mask, IMAGE_SIZE)
    return (mask.astype(np.float32) / 255.0).reshape(
        IMAGE_HEIGHT, IMAGE_WIDTH, 1
    )


def load_test_arrays(dataset_name, input_kind):
    dataset_dir = TEST_ROOT / dataset_name
    image_dir = dataset_dir / "img"
    mask_dir = dataset_dir / "masks"
    names = list_image_names(image_dir)
    validate_image_mask_pairs(image_dir, mask_dir, names)

    images = np.zeros(
        (len(names), IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32
    )
    masks = np.zeros(
        (len(names), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32
    )
    reader = read_raw_image if input_kind == "raw" else read_fmm_image
    for index, name in enumerate(names):
        images[index] = reader(image_dir / name)
        masks[index] = read_ground_truth(mask_dir / name)
    return images, masks, tuple(names)
'''


WEIGHT_CODE = r'''# ============================================================
# 4. 严格查找并完整加载Raw/FMM权重
# ============================================================
WEIGHT_MODEL_DIR = (
    PROJECT_ROOT / "best weight" / "2.different mode" / MODEL_NAME
)


def resolve_one_weight(standard_path, weight_dir, label):
    """优先使用标准文件名，否则只接受目录中唯一的历史H5权重。"""
    if standard_path.is_file():
        return standard_path
    candidates = sorted(path for path in weight_dir.glob("*.h5") if path.is_file())
    if len(candidates) == 1:
        print(f"{label}: using historical weight {candidates[0].resolve()}")
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No {label} weight found. Standard path: {standard_path}; "
            f"folder: {weight_dir}"
        )
    candidate_text = "\n".join(str(path.resolve()) for path in candidates)
    raise RuntimeError(
        f"Multiple {label} H5 weights found; selection is ambiguous:\n{candidate_text}"
    )


def resolve_weights():
    raw_dir = WEIGHT_MODEL_DIR / "img"
    fmm_dir = WEIGHT_MODEL_DIR / "FMM+percentile=99.5"
    raw_weight = resolve_one_weight(
        raw_dir / f"{MODEL_NAME}_img_bestweight.h5", raw_dir, "Raw"
    )
    fmm_weight = resolve_one_weight(
        fmm_dir / f"{MODEL_NAME}_99.5+FMM_bestweight.h5", fmm_dir, "FMM"
    )
    return raw_weight, fmm_weight


def load_weights_strictly(model, weight_path):
    """按拓扑完整加载；模型与权重形状不一致时立即停止，不静默跳过。"""
    try:
        model.load_weights(str(weight_path))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot fully load {MODEL_NAME} weights: {weight_path}"
        ) from exc


tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(SEED)
SEGMENTATION_MODEL = build_model()
RAW_WEIGHT, FMM_WEIGHT = resolve_weights()

print("Raw weight:", RAW_WEIGHT.resolve())
print("FMM weight:", FMM_WEIGHT.resolve())
print("Model parameters:", SEGMENTATION_MODEL.count_params())
'''


METRIC_CODE = r'''# ============================================================
# 5. 逐图重叠、边界与错误类型指标
# ============================================================
METRIC_NAMES = (
    "soft_dice",
    "soft_iou",
    "hard_dice",
    "hard_iou",
    "hd95",
    "boundary_iou",
    "precision",
    "recall",
    "fp_image_ratio",
    "fn_gt_ratio",
)


def safe_ratio(numerator, denominator, default=0.0):
    return float(numerator / denominator) if denominator else float(default)


def per_image_soft_metrics(y_true, y_pred, smooth=1e-7):
    axes = tuple(range(1, y_true.ndim))
    intersection = np.sum(y_true * y_pred, axis=axes, dtype=np.float64)
    true_sum = np.sum(y_true, axis=axes, dtype=np.float64)
    pred_sum = np.sum(y_pred, axis=axes, dtype=np.float64)
    return {
        "soft_dice": (2.0 * intersection + smooth) / (true_sum + pred_sum + smooth),
        "soft_iou": (intersection + smooth)
        / (true_sum + pred_sum - intersection + smooth),
    }


def mask_surface(mask):
    mask = mask.astype(np.uint8)
    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    eroded = cv2.erode(
        padded, np.ones((3, 3), np.uint8), iterations=1
    )[1:-1, 1:-1]
    return mask.astype(bool) & ~eroded.astype(bool)


def hd95_for_image(true_mask, pred_mask):
    true_surface = mask_surface(true_mask)
    pred_surface = mask_surface(pred_mask)
    if not true_surface.any() and not pred_surface.any():
        return 0.0
    if not true_surface.any() or not pred_surface.any():
        return float(math.hypot(IMAGE_HEIGHT, IMAGE_WIDTH))

    distance_to_true = cv2.distanceTransform(
        (~true_surface).astype(np.uint8), cv2.DIST_L2, 5
    )
    distance_to_pred = cv2.distanceTransform(
        (~pred_surface).astype(np.uint8), cv2.DIST_L2, 5
    )
    distances = np.concatenate(
        [distance_to_true[pred_surface], distance_to_pred[true_surface]]
    )
    return float(np.percentile(distances, 95))


def boundary_mask(mask):
    width = max(
        1,
        int(round(BOUNDARY_DILATION_RATIO * math.hypot(IMAGE_HEIGHT, IMAGE_WIDTH))),
    )
    mask = mask.astype(np.uint8)
    padded = np.pad(mask, width, mode="constant", constant_values=0)
    eroded = cv2.erode(
        padded, np.ones((3, 3), np.uint8), iterations=width
    )[width:-width, width:-width]
    return mask.astype(bool) & ~eroded.astype(bool)


def per_image_detailed_metrics(y_true_soft, y_pred_probability):
    if y_true_soft.shape != y_pred_probability.shape:
        raise ValueError(
            f"Prediction shape mismatch: {y_pred_probability.shape} != {y_true_soft.shape}"
        )
    if not np.isfinite(y_pred_probability).all():
        raise FloatingPointError("Predictions contain NaN or Inf")

    values = per_image_soft_metrics(y_true_soft, y_pred_probability)
    for metric_name in METRIC_NAMES[2:]:
        values[metric_name] = []

    y_true = y_true_soft >= PREDICTION_THRESHOLD
    y_pred = y_pred_probability >= PREDICTION_THRESHOLD
    for index in range(len(y_true)):
        true_mask = y_true[index, ..., 0]
        pred_mask = y_pred[index, ..., 0]
        tp = int(np.logical_and(true_mask, pred_mask).sum())
        fp = int(np.logical_and(~true_mask, pred_mask).sum())
        fn = int(np.logical_and(true_mask, ~pred_mask).sum())

        true_boundary = boundary_mask(true_mask)
        pred_boundary = boundary_mask(pred_mask)
        boundary_intersection = np.logical_and(true_boundary, pred_boundary).sum()
        boundary_union = np.logical_or(true_boundary, pred_boundary).sum()

        values["hard_dice"].append(
            safe_ratio(2 * tp, 2 * tp + fp + fn, default=1.0)
        )
        values["hard_iou"].append(
            safe_ratio(tp, tp + fp + fn, default=1.0)
        )
        values["hd95"].append(hd95_for_image(true_mask, pred_mask))
        values["boundary_iou"].append(
            safe_ratio(boundary_intersection, boundary_union, default=1.0)
        )
        values["precision"].append(
            safe_ratio(tp, tp + fp, default=1.0 if not true_mask.any() else 0.0)
        )
        values["recall"].append(safe_ratio(tp, tp + fn, default=1.0))
        values["fp_image_ratio"].append(safe_ratio(fp, true_mask.size))
        values["fn_gt_ratio"].append(safe_ratio(fn, tp + fn))

    for metric_name in METRIC_NAMES[2:]:
        values[metric_name] = np.asarray(values[metric_name], dtype=np.float64)
    return values
'''


ANALYSIS_CODE = r'''# ============================================================
# 6. RR/RF/FR/FF四路推理与逐图坏案例表
# ============================================================
PATHWAYS = {
    "RR": {"input": "raw", "weight": RAW_WEIGHT},
    "RF": {"input": "fmm", "weight": RAW_WEIGHT},
    "FR": {"input": "raw", "weight": FMM_WEIGHT},
    "FF": {"input": "fmm", "weight": FMM_WEIGHT},
}


def evaluate_pathway(dataset_name, pathway_name):
    pathway = PATHWAYS[pathway_name]
    images, masks, names = load_test_arrays(dataset_name, pathway["input"])
    load_weights_strictly(SEGMENTATION_MODEL, pathway["weight"])
    predictions = np.asarray(
        SEGMENTATION_MODEL.predict(
            images, batch_size=INFERENCE_BATCH_SIZE, verbose=0
        )
    )
    if predictions.shape != masks.shape:
        raise ValueError(
            f"{dataset_name}/{pathway_name}: {predictions.shape} != {masks.shape}"
        )
    metrics = per_image_detailed_metrics(masks, predictions)
    aggregate = aggregate_soft_metrics(masks, predictions)
    del images, predictions
    gc.collect()
    return metrics, aggregate, names, masks


def highlight_attributes(dataset_name, names, masks):
    image_dir = TEST_ROOT / dataset_name / "img"
    rows = []
    for index, image_name in enumerate(names):
        image_path = image_dir / image_name
        raw_image = read_raw_image(image_path)
        fmm_image = read_fmm_image(image_path)
        # 此mask只用于与GT对齐后计算面积/位置关系，不参与FMM修复。
        # 因此使用最近邻插值保持二值标签；真正修复函数内仍严格使用INTER_CUBIC。
        highlight = cv2.resize(
            create_highlight_mask(image_path),
            IMAGE_SIZE,
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        ground_truth = masks[index, ..., 0] >= 0.5
        boundary = boundary_mask(ground_truth)

        highlight_pixels = int(highlight.sum())
        polyp_pixels = int(ground_truth.sum())
        polyp_overlap = int(np.logical_and(highlight, ground_truth).sum())
        boundary_overlap = int(np.logical_and(highlight, boundary).sum())
        rgb_change = np.abs(raw_image - fmm_image)

        rows.append({
            "highlight_area_ratio": safe_ratio(highlight_pixels, highlight.size),
            "highlight_polyp_overlap_ratio": safe_ratio(
                polyp_overlap, highlight_pixels
            ),
            "polyp_covered_by_highlight_ratio": safe_ratio(
                polyp_overlap, polyp_pixels
            ),
            "highlight_boundary_overlap_ratio": safe_ratio(
                boundary_overlap, highlight_pixels
            ),
            "mean_abs_rgb_change": float(rgb_change.mean()),
            "highlight_region_rgb_change": float(
                rgb_change[highlight].mean() if highlight_pixels else 0.0
            ),
        })
    return rows


def validate_against_reference(dataset_name, aggregates):
    rows = []
    for pathway_name in ("RR", "FF"):
        for metric_name in ("Dice", "IoU"):
            observed = float(aggregates[pathway_name][metric_name])
            expected = float(
                REFERENCE_RESULTS[dataset_name][pathway_name][metric_name]
            )
            absolute_error = abs(observed - expected)
            passed = absolute_error <= REFERENCE_TOLERANCE
            rows.append({
                "model": MODEL_NAME,
                "dataset": dataset_name,
                "pathway": pathway_name,
                "metric": metric_name,
                "observed": observed,
                "reference": expected,
                "absolute_error": absolute_error,
                "tolerance": REFERENCE_TOLERANCE,
                "passed": passed,
            })
            if not passed:
                raise AssertionError(
                    f"Reference mismatch for {dataset_name}/{pathway_name}/{metric_name}: "
                    f"{observed} vs {expected}"
                )
    return rows


def analyze_dataset(dataset_name):
    pathway_metrics = {}
    aggregates = {}
    reference_names = None
    reference_masks = None

    for pathway_name in PATHWAYS:
        print(f"  {dataset_name}: {pathway_name}")
        metrics, aggregate, names, masks = evaluate_pathway(
            dataset_name, pathway_name
        )
        if reference_names is None:
            reference_names = names
            reference_masks = masks
        elif names != reference_names:
            raise ValueError(f"{dataset_name}: image order differs across pathways")
        else:
            del masks
        pathway_metrics[pathway_name] = metrics
        aggregates[pathway_name] = aggregate

    attributes = highlight_attributes(
        dataset_name, reference_names, reference_masks
    )
    image_rows = []
    for index, image_name in enumerate(reference_names):
        row = {
            "model": MODEL_NAME,
            "dataset": dataset_name,
            "image_name": image_name,
            **attributes[index],
        }
        for pathway_name in PATHWAYS:
            for metric_name in METRIC_NAMES:
                row[f"{pathway_name}_{metric_name}"] = float(
                    pathway_metrics[pathway_name][metric_name][index]
                )
        for metric_name in METRIC_NAMES:
            rr = row[f"RR_{metric_name}"]
            rf = row[f"RF_{metric_name}"]
            fr = row[f"FR_{metric_name}"]
            ff = row[f"FF_{metric_name}"]
            row[f"delta_main_{metric_name}"] = ff - rr
            row[f"delta_input_raw_weight_{metric_name}"] = rf - rr
            row[f"delta_weight_raw_input_{metric_name}"] = fr - rr
            row[f"delta_adaptation_fmm_input_{metric_name}"] = ff - rf
        row["is_bad_case"] = row["delta_main_soft_dice"] < 0.0
        image_rows.append(row)

    image_table = pd.DataFrame(image_rows)
    summary = {
        "model": MODEL_NAME,
        "dataset": dataset_name,
        "N": len(image_table),
        "RR_aggregate_soft_dice": aggregates["RR"]["Dice"],
        "FF_aggregate_soft_dice": aggregates["FF"]["Dice"],
        "delta_aggregate_soft_dice": (
            aggregates["FF"]["Dice"] - aggregates["RR"]["Dice"]
        ),
        "RR_aggregate_soft_iou": aggregates["RR"]["IoU"],
        "FF_aggregate_soft_iou": aggregates["FF"]["IoU"],
        "delta_aggregate_soft_iou": (
            aggregates["FF"]["IoU"] - aggregates["RR"]["IoU"]
        ),
        "mean_delta_soft_dice": image_table["delta_main_soft_dice"].mean(),
        "median_delta_soft_dice": image_table["delta_main_soft_dice"].median(),
        "bad_case_count": int(image_table["is_bad_case"].sum()),
        "bad_case_rate": float(image_table["is_bad_case"].mean()),
        "degraded_by_aggregate_dice": bool(
            aggregates["FF"]["Dice"] < aggregates["RR"]["Dice"]
        ),
    }
    validation = validate_against_reference(dataset_name, aggregates)
    del reference_masks
    gc.collect()
    return image_table, summary, validation
'''


EXPORT_CODE = r'''# ============================================================
# 7. 运行五个测试集并导出CSV/Excel结果
# ============================================================
image_tables = []
summary_rows = []
validation_rows = []

for dataset_name in TEST_DATASETS:
    print(f"Analyzing {MODEL_NAME}: {dataset_name}")
    image_table, summary, validation = analyze_dataset(dataset_name)
    image_tables.append(image_table)
    summary_rows.append(summary)
    validation_rows.extend(validation)

PER_IMAGE_TABLE = pd.concat(image_tables, ignore_index=True)
DATASET_SUMMARY_TABLE = pd.DataFrame(summary_rows)
VALIDATION_TABLE = pd.DataFrame(validation_rows)

BAD_CASE_TOP5_TABLE = (
    PER_IMAGE_TABLE
    .sort_values(["dataset", "delta_main_soft_dice", "image_name"])
    .groupby("dataset", sort=False, group_keys=False)
    .head(TOP_K)
    .reset_index(drop=True)
)

correlation_features = (
    "highlight_area_ratio",
    "highlight_polyp_overlap_ratio",
    "polyp_covered_by_highlight_ratio",
    "highlight_boundary_overlap_ratio",
    "mean_abs_rgb_change",
    "highlight_region_rgb_change",
)
correlation_rows = []
groups = [
    (dataset_name, PER_IMAGE_TABLE[PER_IMAGE_TABLE["dataset"] == dataset_name])
    for dataset_name in TEST_DATASETS
]
groups.append(("ALL", PER_IMAGE_TABLE))

for group_name, group_table in groups:
    delta = group_table["delta_main_soft_dice"].to_numpy(dtype=np.float64)
    for feature_name in correlation_features:
        feature = group_table[feature_name].to_numpy(dtype=np.float64)
        if np.unique(feature).size < 2 or np.unique(delta).size < 2:
            rho, p_value = np.nan, np.nan
        else:
            test_result = spearmanr(feature, delta, nan_policy="raise")
            rho_value = (
                test_result.statistic
                if hasattr(test_result, "statistic")
                else test_result.correlation
            )
            rho = float(rho_value)
            p_value = float(test_result.pvalue)
        correlation_rows.append({
            "model": MODEL_NAME,
            "dataset": group_name,
            "feature": feature_name,
            "N": len(group_table),
            "spearman_rho": rho,
            "p_value": p_value,
        })

CORRELATION_TABLE = pd.DataFrame(correlation_rows)

PER_IMAGE_CSV = RESULT_DIR / "per_image_metrics.csv"
TOP5_CSV = RESULT_DIR / "bad_case_top5.csv"
SUMMARY_XLSX = RESULT_DIR / "dataset_summary.xlsx"

PER_IMAGE_TABLE.to_csv(PER_IMAGE_CSV, index=False, encoding="utf-8-sig")
BAD_CASE_TOP5_TABLE.to_csv(TOP5_CSV, index=False, encoding="utf-8-sig")
with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl") as writer:
    DATASET_SUMMARY_TABLE.to_excel(
        writer, sheet_name="Dataset Summary", index=False
    )
    BAD_CASE_TOP5_TABLE.to_excel(
        writer, sheet_name="Bad Cases Top5", index=False
    )
    CORRELATION_TABLE.to_excel(
        writer, sheet_name="Correlations", index=False
    )
    VALIDATION_TABLE.to_excel(
        writer, sheet_name="Validation", index=False
    )

display(DATASET_SUMMARY_TABLE.round(4))
display(BAD_CASE_TOP5_TABLE[[
    "dataset",
    "image_name",
    "RR_soft_dice",
    "RF_soft_dice",
    "FR_soft_dice",
    "FF_soft_dice",
    "delta_main_soft_dice",
    "highlight_boundary_overlap_ratio",
]].round(4))
'''


FIGURE_HELPER_CODE = r'''# ============================================================
# 8. 坏案例可视化辅助函数
# ============================================================
def safe_file_stem(value):
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(value)).strip("._")


def confusion_map(prediction, ground_truth):
    """错误图颜色：TP为绿色、FP为红色、FN为蓝色、背景为深灰色。"""
    predicted = np.asarray(prediction).squeeze() >= PREDICTION_THRESHOLD
    truth = np.asarray(ground_truth).squeeze() >= 0.5
    canvas = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 18, dtype=np.uint8)
    canvas[np.logical_and(predicted, truth)] = (44, 170, 80)
    canvas[np.logical_and(predicted, ~truth)] = (225, 55, 55)
    canvas[np.logical_and(~predicted, truth)] = (55, 110, 225)
    return canvas


def highlight_overlay(raw_image, highlight_mask):
    canvas = np.clip(raw_image * 255.0, 0, 255).astype(np.uint8).copy()
    yellow = np.array([255, 220, 35], dtype=np.float32)
    canvas[highlight_mask] = (
        0.45 * canvas[highlight_mask].astype(np.float32) + 0.55 * yellow
    ).astype(np.uint8)
    return canvas


def predict_single_case(dataset_name, image_name):
    dataset_dir = TEST_ROOT / dataset_name
    image_path = dataset_dir / "img" / image_name
    mask_path = dataset_dir / "masks" / image_name
    raw_image = read_raw_image(image_path)
    fmm_image = read_fmm_image(image_path)
    ground_truth = read_ground_truth(mask_path)
    paired_inputs = np.stack([raw_image, fmm_image], axis=0)

    load_weights_strictly(SEGMENTATION_MODEL, RAW_WEIGHT)
    raw_weight_predictions = np.asarray(
        SEGMENTATION_MODEL.predict(paired_inputs, batch_size=2, verbose=0)
    )
    load_weights_strictly(SEGMENTATION_MODEL, FMM_WEIGHT)
    fmm_weight_predictions = np.asarray(
        SEGMENTATION_MODEL.predict(paired_inputs, batch_size=2, verbose=0)
    )

    # 此处最近邻缩放只用于案例图中的高光mask显示，不参与FMM修复。
    highlight = cv2.resize(
        create_highlight_mask(image_path),
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    return {
        "raw": raw_image,
        "fmm": fmm_image,
        "ground_truth": ground_truth,
        "highlight": highlight,
        "predictions": {
            "RR": raw_weight_predictions[0],
            "RF": raw_weight_predictions[1],
            "FR": fmm_weight_predictions[0],
            "FF": fmm_weight_predictions[1],
        },
    }


def save_case_figure(row, case, output_path):
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    top_panels = (
        (case["raw"], "Raw image"),
        (case["fmm"], "99.5% + FMM image"),
        (highlight_overlay(case["raw"], case["highlight"]), "Highlight mask"),
        (case["ground_truth"][..., 0], "Ground truth"),
    )
    for axis, (panel, title) in zip(axes[0], top_panels):
        axis.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")

    for axis, pathway_name in zip(axes[1], PATHWAYS):
        dice_column = f"{pathway_name}_soft_dice"
        axis.imshow(
            confusion_map(
                case["predictions"][pathway_name], case["ground_truth"]
            )
        )
        axis.set_title(
            f"{pathway_name} error map\nSoft Dice={row[dice_column]:.4f}"
        )
        axis.axis("off")

    figure.suptitle(
        f"{MODEL_NAME} | {row['dataset']} | {row['image_name']} | "
        f"Delta Dice={row['delta_main_soft_dice']:.4f}\n"
        "TP=green, FP=red, FN=blue",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
'''


FIGURE_RUN_CODE = r'''# ============================================================
# 9. 生成Dice变化分布、逐例2x4图和总体案例图
# ============================================================
distribution_path = FIGURE_DIR / "delta_distribution.png"
figure, axis = plt.subplots(figsize=(11, 5.8))
delta_groups = [
    PER_IMAGE_TABLE.loc[
        PER_IMAGE_TABLE["dataset"] == dataset_name,
        "delta_main_soft_dice",
    ].to_numpy()
    for dataset_name in TEST_DATASETS
]
boxplot = axis.boxplot(
    delta_groups,
    labels=TEST_DATASETS,
    patch_artist=True,
    showfliers=False,
)
for box in boxplot["boxes"]:
    box.set_facecolor("#4C78A8")
    box.set_alpha(0.72)
axis.axhline(0.0, color="#222222", linewidth=1.2, linestyle="--")
axis.set_ylabel("Per-image Soft Dice delta (FF - RR)")
axis.set_title(f"{MODEL_NAME}: per-image performance change")
axis.tick_params(axis="x", rotation=18)
axis.grid(axis="y", alpha=0.25)
figure.tight_layout()
figure.savefig(distribution_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(figure)

degraded_datasets = set(
    DATASET_SUMMARY_TABLE.loc[
        DATASET_SUMMARY_TABLE["degraded_by_aggregate_dice"], "dataset"
    ]
)
CASE_ROWS = BAD_CASE_TOP5_TABLE[
    BAD_CASE_TOP5_TABLE["dataset"].isin(degraded_datasets)
].copy()
if CASE_ROWS.empty:
    CASE_ROWS = PER_IMAGE_TABLE.sort_values("delta_main_soft_dice").head(TOP_K)

case_cache = {}
generated_rows = []
for _, row in CASE_ROWS.iterrows():
    key = (row["dataset"], row["image_name"])
    case = predict_single_case(*key)
    case_cache[key] = case
    dataset_dir = CASE_FIGURE_DIR / safe_file_stem(row["dataset"])
    dataset_dir.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"{safe_file_stem(Path(row['image_name']).stem)}"
        f"_delta_{row['delta_main_soft_dice']:+.4f}.png"
    )
    output_path = dataset_dir / file_name
    save_case_figure(row, case, output_path)
    generated_rows.append({
        "dataset": row["dataset"],
        "image_name": row["image_name"],
        "figure_path": str(output_path.resolve()),
    })

GENERATED_CASE_TABLE = pd.DataFrame(generated_rows)

overview_rows = CASE_ROWS.sort_values("delta_main_soft_dice").head(3)
overview_path = FIGURE_DIR / "bad_case_overview.png"
figure, axes = plt.subplots(
    len(overview_rows),
    4,
    figsize=(14, max(3.6, 3.6 * len(overview_rows))),
    squeeze=False,
)
for row_index, (_, row) in enumerate(overview_rows.iterrows()):
    key = (row["dataset"], row["image_name"])
    case = case_cache.get(key) or predict_single_case(*key)
    panels = (
        (case["raw"], "Raw"),
        (case["fmm"], "FMM"),
        (
            confusion_map(case["predictions"]["RR"], case["ground_truth"]),
            f"RR Dice={row['RR_soft_dice']:.3f}",
        ),
        (
            confusion_map(case["predictions"]["FF"], case["ground_truth"]),
            f"FF Dice={row['FF_soft_dice']:.3f}",
        ),
    )
    for axis, (panel, title) in zip(axes[row_index], panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    axes[row_index, 0].set_ylabel(
        f"{row['dataset']}\n{row['image_name']}\n"
        f"Delta={row['delta_main_soft_dice']:.3f}"
    )

figure.suptitle(
    f"{MODEL_NAME}: representative degradation cases\n"
    "TP=green, FP=red, FN=blue",
    fontsize=13,
)
figure.tight_layout(rect=(0, 0, 1, 0.95))
figure.savefig(overview_path, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(figure)

print("Distribution figure:", distribution_path.resolve())
print("Individual case figures:", len(GENERATED_CASE_TABLE))
print("Overview figure:", overview_path.resolve())
'''


VERIFY_CODE = r'''# ============================================================
# 10. 最终一致性检查
# ============================================================
expected_count = int(DATASET_SUMMARY_TABLE["N"].sum())
if len(PER_IMAGE_TABLE) != expected_count:
    raise AssertionError(
        f"Per-image row count mismatch: {len(PER_IMAGE_TABLE)} != {expected_count}"
    )
if not VALIDATION_TABLE["passed"].all():
    raise AssertionError("RR/FF aggregate metrics do not match reference results")
if PER_IMAGE_TABLE["image_name"].isna().any():
    raise AssertionError("Missing image names in per-image results")
numeric_values = PER_IMAGE_TABLE.select_dtypes(include=[np.number]).to_numpy()
if not np.isfinite(numeric_values).all():
    raise AssertionError("NaN or Inf found in per-image results")

required_files = (
    PER_IMAGE_CSV,
    TOP5_CSV,
    SUMMARY_XLSX,
    distribution_path,
    overview_path,
)
missing_files = [str(path) for path in required_files if not path.is_file()]
if missing_files:
    raise FileNotFoundError("Missing outputs:\n" + "\n".join(missing_files))

print("All standalone bad-case analysis checks passed.")
print("Per-image rows:", len(PER_IMAGE_TABLE))
print("Bad cases:", int(PER_IMAGE_TABLE["is_bad_case"].sum()))
print("Results:", RESULT_DIR.resolve())
'''


def source_lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(text)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


def render_template(template: str, spec: dict) -> str:
    return (
        template.replace("__MODEL_NAME__", spec["model_name"])
        .replace("__ARCHITECTURE__", spec["architecture"])
        .replace("__REFERENCE_RESULTS__", pformat(spec["reference"], width=88))
    )


def build_notebook(spec: dict) -> dict:
    cells = [
        markdown_cell(render_template(INTRO_TEMPLATE, spec)),
        code_cell(render_template(CONFIG_TEMPLATE, spec)),
        code_cell(MODEL_CODE),
        code_cell(DATA_CODE),
        code_cell(WEIGHT_CODE),
        code_cell(METRIC_CODE),
        code_cell(ANALYSIS_CODE),
        code_cell(EXPORT_CODE),
        code_cell(FIGURE_HELPER_CODE),
        code_cell(FIGURE_RUN_CODE),
        code_cell(VERIFY_CODE),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.8.18"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def validate_notebook(notebook: dict) -> None:
    notebook_text = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    forbidden_tokens = (
        "RUN_TRAINING",
        "Train.txt",
        "Val.txt",
        "ALL_RESULTS",
        "Raw-vs-FMM-Train-Inference.ipynb",
    )
    found_forbidden = [token for token in forbidden_tokens if token in notebook_text]
    if found_forbidden:
        raise RuntimeError(
            "Standalone notebook contains external dependencies: "
            + ", ".join(found_forbidden)
        )

    required_tokens = (
        '"RR"',
        '"RF"',
        '"FR"',
        '"FF"',
        "def create_highlight_mask(image_path, percentile=PERCENTILE)",
        "threshold = np.percentile(gray, percentile)",
        "fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC",
        "resized_source, resized_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA",
        '".jpg", restored, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]',
        "restored = restored[..., ::-1]",
        "validate_fmm_configuration()",
        "per_image_metrics.csv",
        "dataset_summary.xlsx",
        "bad_case_top5.csv",
        "delta_distribution.png",
        "bad_case_overview.png",
        "bad_case_analysis_results",
    )
    missing = [token for token in required_tokens if token not in notebook_text]
    if missing:
        raise RuntimeError("Missing standalone content: " + ", ".join(missing))

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        compile("".join(cell["source"]), f"cell_{index}", "exec")
        if cell["outputs"] or cell["execution_count"] is not None:
            raise RuntimeError(f"Cell {index} contains stale execution output")


def write_notebook(spec: dict) -> Path:
    notebook = build_notebook(spec)
    validate_notebook(notebook)
    output_path = OUTPUT_DIR / spec["output_name"]
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(copy.deepcopy(notebook), handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    return output_path


def main() -> None:
    for spec in NOTEBOOK_SPECS:
        print(f"Generated standalone notebook: {write_notebook(spec)}")


if __name__ == "__main__":
    main()
