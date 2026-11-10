# Sleep Apnea Detection from Single-Lead ECG

## Overview
This repository contains a comparative machine learning framework for minute-by-minute sleep apnea detection using single-lead ECG signals. The project evaluates time-domain HRV, frequency-domain HRV, and wavelet-based features across Support Vector Machines (SVM), Random Forest, and Feedforward Neural Network classifiers.

Aligned with **UN Sustainable Development Goal 3 (Good Health and Well-being)**, specifically Targets 3.4 and 3.8, this project aims to provide an accurate, interpretable, low-resource computational pipeline suitable for wearable screening and accessible healthcare diagnostics.

---

## Workspace & Directory Structure

```
SleepApnea_ECG/
├── data/
│   ├── apnea-ecg/          % PhysioNet Apnea-ECG (.dat, .hea, .apn, .qrs)
│   ├── slpdb/              % MIT-BIH PSG
│   └── ucddb/              % UCD Sleep Apnea
├── code/
│   ├── 01_preprocessing/   % Bandpass filter, baseline removal, Pan-Tompkins R-peak detector
│   ├── 02_features/        % Time-domain HRV, Freq-domain HRV, Wavelet & EDR extraction
│   ├── 03_classification/  % SVM, Random Forest, Feedforward NN training scripts
│   ├── 04_evaluation/      % Cross-validation, ablation study, statistical tests, external validation
│   └── utils/              % Environment setup, dataset checkers, download utilities
├── results/
│   ├── figures/            % 300 DPI publication-quality figures
│   ├── tables/             % CSV & MAT exported performance tables
│   └── models/             % Trained classification model files
├── paper/                  % Research paper drafts, data understanding notes, references
└── README.md               % Project documentation & sitemap
```

---

## Requirements & MATLAB Toolboxes
- **MATLAB Version:** R2023b or R2024a
- **Required Toolboxes:**
  - Signal Processing Toolbox (`designfilt`, `filtfilt`, `findpeaks`, `pwelch`)
  - Wavelet Toolbox (`wavedec`, `appcoef`, `detcoef`, `wentropy`, `wdenoise`)
  - Statistics and Machine Learning Toolbox (`fitcsvm`, `TreeBagger`, `fitcnet`, `relieff`, `confusionmat`, `perfcurve`)
  - WFDB Toolbox for MATLAB (for reading PhysioNet signals & annotations)

---

## Phase Execution Roadmap
1. **Phase 0:** Environment setup, WFDB installation & dataset acquisition (`code/utils/`)
2. **Phase 1:** Signal filtering, baseline removal & Pan-Tompkins R-peak validation (`code/01_preprocessing/`)
3. **Phase 2:** Per-minute feature extraction, normalization & correlation pruning (`code/02_features/`)
4. **Phase 3:** Classifier training with hyperparameter optimization (`code/03_classification/`)
5. **Phase 4:** Comparative evaluation, feature ablation, McNemar statistical test & external validation (`code/04_evaluation/`)
6. **Phase 5:** Research manuscript drafting & paper submission package (`paper/`)
