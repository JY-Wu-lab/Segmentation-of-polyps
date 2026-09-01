# Supplementary Code

This repository contains the executed notebooks and archived outputs supporting the manuscript **"Artifact-Aware Restoration for Improving Robustness and Generalization in Polyp Segmentation."**

The proposed preprocessing condition, **FMM-99.5**, generates an image-specific highlight mask using the 99.5th grayscale-intensity percentile and applies OpenCV TELEA inpainting with a radius of 10 pixels.

## Directory structure

| Directory | Description |
|---|---|
| `01_parameter_selection` | Training-only percentile and FMM-radius selection using repeated five-fold cross-validation. |
| `02_multi_architecture_evaluation-Raw-vs-FMM-Wilcoxon-PairedT` | Raw-versus-FMM training, inference, segmentation metrics, and paired statistical tests for ten model configurations. |
| `03_runtime_analysis` | Runtime benchmarking of FMM, Navier-Stokes, Partial Convolution, and DeepFill v2. |
| `04_color_fidelity_analysis` | Color-fidelity audit of the DeepFill v2 outputs. |
| `05_feature_interference_analysis` | Controlled feature-interference analyses for FPN+V3 and LinkNet+V3. |
| `06_cross_dataset_evaluation` | Source-held-out evaluation using CVC-ColonDB as the source dataset and four external target datasets. |

## Usage

1. Open the required notebook in Jupyter Notebook or JupyterLab.
2. Set the dataset, project, output, and checkpoint paths in the configuration cell near the beginning of the notebook.
3. Use the dataset structure described in the notebook or in the README supplied with the multi-architecture experiment.
4. Restart the kernel and run all cells from top to bottom.

The notebooks retain their executed outputs so that the reported calculations, tables, and figures can be inspected without rerunning every experiment. Re-executing training or inference requires the corresponding public datasets, model dependencies, and pretrained or archived checkpoints.

## Experimental notes

- Raw and FMM-99.5 conditions use identical data indices within each paired comparison.
- The main multi-architecture experiment uses a fixed image-level split and random seed 42.
- Restoration-parameter selection is performed exclusively on the training pool using repeated five-fold cross-validation with seeds 42, 123, and 2026.
- Descriptive Dice and IoU are calculated over consecutive metric groups of up to 32 images. Paired statistical tests use per-image soft Dice and soft IoU values matched by filename.
- Detailed metric definitions, statistical procedures, software environments, and dataset descriptions are provided in the manuscript and Supplementary Material.

## Data and checkpoints

The public colonoscopy datasets are not redistributed in this repository. Users should obtain them from their original sources and update the local paths in each notebook. Model checkpoints are distributed separately through the data/code availability link reported with the manuscript.

## Citation

If this code is useful in your research, please cite the associated Pattern Recognition manuscript.
