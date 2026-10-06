# WESAD wrist-PPG stress classification: beat-shape and wavelet features versus interval features

Analysis code and output files for the manuscript **"WESAD wrist-PPG stress classification: beat-shape and wavelet features versus interval features"** (Mahima Thapa, 2026).

The study asks one narrow question on one public dataset: on the same windows and the same held-out subjects, do beat-shape and wavelet features add information to heart rate and interval features for classifying baseline versus stress from wrist photoplethysmography (PPG)?

## Summary of findings

| Topic | Result |
|---|---|
| Data | WESAD, 15 subjects, Empatica E4 blood volume pulse (64 Hz), baseline versus Trier Social Stress Test, 60 s windows with a 30 s step |
| Stage 1 windows | 620 quality-controlled windows (514 baseline, 106 stress; stress data from 11 subjects) |
| Paired contrasts | Heart rate was the only feature with a significant change after Holm correction (+24.4 bpm, Holm p = 0.035) |
| Dicrotic notch features | Notch-based features (DR, RI, AR) were unavailable in every stress window; the cause was not tested |
| Classification | Under leave-one-subject-out validation, the 95% subject-bootstrap intervals of interval-only and all-feature sets overlap |
| Stage 2 | Exploratory and post hoc, run once with fixed parameters (602 windows; stress data from 9 subjects). It does not provide confirmatory evidence |

Read the manuscript for the full results, limitations and exact numbers.

## Repository layout

```
code/
  wesad_ppg_pipeline.py          Stage 1: windows, quality control, features, statistics, models
  wesad_ppg_pipeline2.py         Stage 2 (exploratory): max dP/dt timing, APPG features, within-subject z-scoring
  wesad_ppg_pipeline2_extra.py   Stage 2 decomposition (exploratory): which feature family carries the gain
results/                         Stage 1 outputs (CSV tables, windows.csv, figures)
results_p2/                      Stage 2 outputs (p2_*.csv, p2_run_log.json, p2_extra_*.csv)
paper/                           Manuscript (PDF, LaTeX source, figures)
```

The WESAD data are not included. Download them from the original authors (see below).

## Reproduce the analysis

1. Download WESAD (about 2 GB) from the UCI Machine Learning Repository and unzip it so that the folders look like `WESAD/S2/S2.pkl`, `WESAD/S3/S3.pkl`, and so on. Subjects S1 and S12 do not exist in the release.
2. Install Python 3.10 or newer and the packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the three scripts in order from inside `code/` (replace the data path with your own):
   ```
   python wesad_ppg_pipeline.py --data /path/to/WESAD --out ../results
   python wesad_ppg_pipeline2.py --data /path/to/WESAD --out ../results_p2 --p1_windows ../results/windows.csv
   python wesad_ppg_pipeline2_extra.py --p2_dir ../results_p2
   ```
   The third script reads only the CSV files written by the second one and does not need the WESAD data.

Features are cached as `windows.csv`; add `--recompute` to rebuild them. Random seeds are fixed in the scripts, but results can differ slightly across library versions.

## Output files

| File | Content |
|---|---|
| `results/window_counts.csv`, `results/windows.csv` | Window counts and per-window features (Stage 1) |
| `results/table1_features.csv` | Paired contrasts, baseline versus stress |
| `results/table2_models.csv`, `table3_ablation.csv`, `table4_importance.csv` | Classification, feature-family ablation, permutation importance |
| `results_p2/p2_windows.csv`, `p2_window_retention.csv` | Stage 2 window features and retention per subject |
| `results_p2/p2_comparison_matrix.csv`, `p2_per_subject_auc_matrix.csv` | Stage 2 AUC matrix with bootstrap intervals, per-subject AUC |
| `results_p2/p2_extra_matrix.csv`, `p2_extra_per_subject.csv` | Stage 2 decomposition |
| `results_p2/p2_run_log.json` | Parameters of the single Stage 2 run |

## Method notes

- The BVP signal is band-pass filtered (0.5 to 8 Hz, zero phase) and cut into 60 s windows with a 30 s step. Windows are labelled baseline or stress from the 700 Hz label stream.
- Stage 1 features: HR, SDNN, RMSSD, dicrotic-notch ratios (DR, RI, AR) and two Morlet continuous wavelet features.
- Stage 2 (exploratory): beat timing at the maximum rate of rise, second-derivative (APPG) features, and within-subject z-scoring. The z-scoring uses the held-out subject's own unlabelled recording, so it is transductive.
- Evaluation: leave-one-subject-out cross-validation with fixed hyperparameters (Logistic Regression, SVM-RBF, Random Forest, XGBoost) and a subject-level bootstrap for 95% intervals.

## Limitations

Only 11 subjects (Stage 1) and 9 subjects (Stage 2) have stress windows, quality control keeps a selected subset of stress windows, and the chest ECG was not used as a reference. See the Limitations section of the manuscript.

## Citing

If you use this code, please cite the manuscript (see `CITATION.cff`) and the WESAD dataset:

Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. *Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI '18)*, 400-408. https://doi.org/10.1145/3242969.3242985

## Author

Mahima Thapa, Independent Researcher, Kathmandu, Nepal
Email: thapahima077@gmail.com
LinkedIn: https://www.linkedin.com/in/mahimathapa/

## License

Code is released under the MIT License (see `LICENSE`). The WESAD dataset has its own terms; follow those of the original authors.
