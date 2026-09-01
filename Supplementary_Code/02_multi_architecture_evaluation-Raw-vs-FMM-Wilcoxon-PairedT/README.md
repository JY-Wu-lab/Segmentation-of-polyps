# Raw versus FMM-99.5 multi-architecture evaluation

## 1. Overview

This directory contains the ten executed Jupyter Notebooks used for the multi-architecture Raw-versus-FMM-99.5 experiment. The notebooks evaluate the same five test datasets under two paired input conditions:

- **Raw:** original colonoscopy images;
- **FMM-99.5:** images processed using the image-specific 99.5th-percentile highlight mask and OpenCV TELEA inpainting with a radius of 10 pixels.

The archived notebooks retain their executed inference, metric-calculation, statistical-analysis, and visualization outputs. The corresponding Excel files are provided in `statistical_test_results` and `statistical_test_results-保留4四位小数`.

By default, the notebooks use `RUN_TRAINING=False` and load the archived Raw and FMM-99.5 checkpoints. This mode reproduces inference and analysis without retraining the models. Set `RUN_TRAINING=True` only when independent retraining is required and the complete training data and output directories have been configured.

## 2. Included notebooks and manuscript model names

| Manuscript configuration | Notebook |
|---|---|
| U-Net++ | `U-net++-Raw-vs-FMM-Train-Inference.ipynb` |
| U-Net+V3 | `U-net+V3-Raw-vs-FMM-Train-Inference.ipynb` |
| FPN+V3 | `FPN+V3-Raw-vs-FMM-Train-Inference.ipynb` |
| LinkNet+V3 | `Linknet+V3-Raw-vs-FMM-Train-Inference.ipynb` |
| PSPNet+V3 | `PSPnet+V3-Raw-vs-FMM-Train-Inference.ipynb` |
| SegNet-Simple | `SegNet-simple-Raw-vs-FMM-Train-Inference.ipynb` |
| SegNet | `SegNet-VGG-Raw-vs-FMM-Train-Inference.ipynb` |
| PraNet | `PraNet-Raw-vs-FMM-Train-Inference.ipynb` |
| TransUNet | `TransUNet-Raw-vs-FMM-Train-Inference.ipynb` |
| Polyp-PVT | `Polyp-PVT_Raw-vs-FMM_TensorFlow.ipynb` |

The result prefix `SegNet` corresponds to **SegNet-Simple** in the manuscript, whereas `SegNet-VGG` corresponds to **SegNet** in Table 4 and the Supplementary Material.

## 3. Archived environment and training protocol

The multi-architecture experiments were conducted using:

- Python 3.8.18;
- TensorFlow 2.9.0 with `tf.keras`;
- NVIDIA GeForce RTX 3080 GPU;
- random seed 42;
- a fixed image-level split of 2,390 optimization images and 597 validation images, shared by each paired Raw/FMM comparison;
- an input resolution of 384 × 288 pixels;
- Adam optimization;
- a maximum of 300 epochs;
- an equally weighted sum of binary cross-entropy and Jaccard losses;
- validation-IoU checkpoint selection;
- `ReduceLROnPlateau` with factor 0.5 and patience 20;
- no early stopping and no online geometric or photometric augmentation.

The initial learning rate was `1e-3` for all configurations except SegNet-Simple. For SegNet-Simple, the learning rate was linearly scaled to `1.25e-4` for a training batch size of 8 relative to the reference batch size of 64.

## 4. Batch settings

Training batch size, inference batch size, and descriptive metric-group size are separate settings. The forward-pass batch size does not define the reported descriptive Dice/IoU aggregation.

| Configuration | Training batch | Inference batch | Descriptive metric group |
|---|---:|---:|---:|
| U-Net++ | 2 | 32 | up to 32 |
| U-Net+V3 | 8 | 8 | up to 32 |
| FPN+V3 | 8 | 32 | up to 32 |
| LinkNet+V3 | 4 | 32 | up to 32 |
| PSPNet+V3 | 8 | 32 | up to 32 |
| SegNet-Simple | 8 | 8 | up to 32 |
| SegNet | 64 | 32 | up to 32 |
| PraNet | 2 | 32 | up to 32 |
| TransUNet | 2 | 32 | up to 32 |
| Polyp-PVT | 4 | 4 | up to 32 |

## 5. Required project structure

Each notebook contains a `PROJECT_ROOT` configuration variable. Before running a notebook on another computer, replace the archived author-machine path with the local project root. The configured root must provide the following resources:

```text
PROJECT_ROOT/
├── Data/
│   ├── Train/
│   │   ├── img/
│   │   └── masks/
│   └── Test/
│       ├── CVC-300/
│       │   ├── img/
│       │   └── masks/
│       ├── CVC-ClinicDB/
│       │   ├── img/
│       │   └── masks/
│       ├── CVC-ColonDB/
│       │   ├── img/
│       │   └── masks/
│       ├── ETIS-LaribPolypDB/
│       │   ├── img/
│       │   └── masks/
│       └── Kvasir/
│           ├── img/
│           └── masks/
├── code/
│   ├── Train.txt
│   └── Val.txt
└── best weight/
    └── 2.different mode/
        └── <model-specific weight directories>/
```

Raw and FMM-99.5 conditions must use identical training, validation, and test-image indices. Do not change the ordering of test filenames between the paired conditions.

## 6. Strict checkpoint loading

The notebooks use strict, complete checkpoint loading to prevent silent partial initialization:

- `by_name=True` is not used for legacy HDF5 checkpoints. Automatic Keras layer names may change after `clear_session()`, and name-based loading can silently skip parameters.
- Legacy HDF5 checkpoints are loaded by topology and validated for completeness.
- SegNet and SegNet-VGG checkpoints using the Keras 3 `layers/*/vars/*` layout are loaded layer by layer, with checks for layer names, weight counts, and tensor shapes.
- When `RUN_TRAINING=False`, ImageNet weights are not downloaded before loading the complete archived checkpoint. ImageNet initialization is retained when retraining configurations that require it.
- Non-finite predictions are treated as errors. The notebook stops and reports the model, dataset, condition, and number of invalid elements instead of continuing with unreliable metrics.

A successful run should print a strict-loading confirmation. A model–checkpoint mismatch should terminate execution with a diagnostic message.

## 7. Metric definitions and aggregation

Two aggregation schemes are intentionally used because descriptive reporting and paired inference serve different purposes.

### 7.1 Descriptive Dice and IoU

The prediction arrays are retained in fixed test-image order and divided into consecutive metric groups of up to 32 images. Soft Dice and soft IoU are calculated from continuous prediction probabilities by summing the intersection and area terms over all pixels and images within each group. Dataset-level values are the arithmetic means of the group-level scores. A final incomplete group is retained and contributes one group-level value.

Raw and FMM-99.5 use identical filenames, ordering, and group construction. These values correspond to Table 4 of the manuscript and the even-numbered complete-metric tables in the Supplementary Material.

### 7.2 Paired inferential analysis

Soft Dice and soft IoU are also calculated separately for every test image. Raw and FMM-99.5 scores are paired by filename and analyzed using two-sided Wilcoxon signed-rank and paired t tests. Holm correction is applied across the five datasets separately for each model, metric, and statistical-test family.

Because the descriptive and inferential analyses use different evaluation units, their numerical means are complementary but are not required to be identical.

### 7.3 Binary metrics

Sensitivity, specificity, precision, F1, HD95, and Boundary IoU use binary predictions obtained with a threshold of 0.5. Soft Dice/IoU and binary F1 therefore have different definitions and are not expected to be numerically identical.

## 8. Output files

Each notebook generates three Excel workbooks:

1. `<Model>_Raw_vs_FMM_01_Dice_IoU_Summary.xlsx`  
   Descriptive group-level soft Dice and soft IoU.
2. `<Model>_Raw_vs_FMM_02_Complete_Metrics.xlsx`  
   Dice, IoU, sensitivity, specificity, precision, F1, HD95, Boundary IoU, and MAE.
3. `<Model>_Raw_vs_FMM_03_Paired_Statistical_Tests.xlsx`  
   Per-image paired differences, Wilcoxon signed-rank tests, paired t tests, and Holm-adjusted p values.

`statistical_test_results` retains the full exported precision. `statistical_test_results-保留4四位小数` contains the four-decimal presentation copies used for manuscript and supplementary-table checking.

The qualitative examples reuse predictions cached in memory during the same batched forward pass used for metric calculation. The notebooks do not save complete prediction-probability maps as persistent supplementary files.

## 9. Recommended execution procedure

1. Open the required notebook in the archived TensorFlow/Keras environment.
2. Set `PROJECT_ROOT` to the local directory containing `Data`, `code`, and `best weight`.
3. Keep `RUN_TRAINING=False` when reproducing inference from the archived checkpoints.
4. Optionally set `VIS_DATASET` and `VIS_SAMPLE_INDEX` to change the displayed qualitative example.
5. Restart the kernel and run all cells from top to bottom.
6. Confirm that strict checkpoint loading succeeds and that Raw/FMM image filenames and counts match for every dataset.
7. Compare the newly exported workbooks with the supplied files in `statistical_test_results`.

Historical execution outputs may display absolute paths from the original workstation. These paths document the archived run and do not replace the user-configurable `PROJECT_ROOT` in the code. The `.ipynb_checkpoints` directory contains editor-generated duplicate files and is not part of the experimental evidence or required execution path.
