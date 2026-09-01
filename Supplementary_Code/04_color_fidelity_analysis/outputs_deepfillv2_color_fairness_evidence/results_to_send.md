# DeepFill v2 color-fairness results

- Formal full analysis: yes
- Strictly paired images available: 2987
- Images processed in this run: 2987
- Analysis excludes the 99.5% highlight mask and a 3-pixel protection band.
- Representative case selection: manual

## Outside-mask color preservation

| Method | Mean deltaE00 (95% CI) | Median deltaE00 [IQR] | RGB MAE | Mean deltaL* | Mean deltaa* | Mean deltab* | R/B-swap better |
|---|---:|---:|---:|---:|---:|---:|---:|
| 99.5% + FMM | 0.8582 [0.8507, 0.8657] | 0.8227 [0.7309, 0.9230] | 1.2680 | -0.0140 | -0.0350 | -0.0242 | 0/2987 |
| 99.5% + Partial Conv | 0.5842 [0.5666, 0.6018] | 0.7757 [0.1011, 0.9880] | 0.9574 | -0.0793 | -0.0353 | 0.0263 | 0/2987 |
| 99.5% + DeepFill v2 (old) | 0.1706 [0.1687, 0.1725] | 0.1716 [0.1316, 0.2106] | 0.3692 | -0.1165 | 0.0623 | 0.1473 | 0/2987 |
| 99.5% + DeepFill v2 (new) | 0.1706 [0.1687, 0.1725] | 0.1716 [0.1316, 0.2106] | 0.3692 | -0.1165 | 0.0623 | 0.1473 | 0/2987 |

## DeepFill old/new consistency

- Pixel-identical images: 212/2987 (7.10%)
- Mean pixel MAE: 0.00011117 on the 0-255 scale
- Maximum per-image pixel MAE: 0.00049506
- Maximum single-channel absolute difference: 2
- Mean changed-pixel fraction: 0.00031508

## Manually selected representative cases

| Label | Image name | New DeepFill mean deltaE00 |
|---|---|---:|
| Case 1 | `39` | 0.1664 |
| Case 2 | `c1_100h0050 (116)` | 0.1711 |
| Case 3 | `c3_endocv2021_001 (181)` | 0.1469 |

## Interpretation boundary

These measurements test whether the evaluated output pipeline introduces a whole-image color shift outside the repaired region. They do not establish anatomical correctness of synthesized pixels inside the highlight mask.

FMM images are JPEG files while the learning-based outputs are primarily PNG files. Cross-method differences therefore describe the complete stored-image pipelines and should not be interpreted as a pure ranking of inpainting algorithms.

The representative cases in the qualitative figure were manually selected. Quantitative conclusions are based on all paired images, not only these displayed examples.

## Files

- `color_fairness_summary.csv`
- `per_image_color_metrics.csv`
- `deepfill_old_new_consistency.csv`
- `deepfill_old_new_summary.csv`
- `representative_selection.csv`
- `figures/color_fairness_evidence.png`
- `figures/representative_cases_manual.png`
