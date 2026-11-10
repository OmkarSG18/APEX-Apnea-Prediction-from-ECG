# APEX: Apnea Prediction from ECG using Explainable Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MATLAB R2023b/R2024a](https://img.shields.io/badge/MATLAB-R2023b%20%7C%20R2024a-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![PhysioNet Apnea-ECG](https://img.shields.io/badge/Dataset-PhysioNet%20Apnea--ECG-brightgreen.svg)](https://physionet.org/content/apnea-ecg/1.0.0/)
[![UN SDG 3](https://img.shields.io/badge/UN%20SDG-3%20Good%20Health-green.svg)](https://sdgs.un.org/goals/goal3)

A comparative machine learning framework for minute-by-minute sleep apnea detection from single-lead electrocardiogram (ECG) recordings. APEX extracts hybrid time-domain, frequency-domain, wavelet-based, and ECG-derived respiration (EDR) features to evaluate Support Vector Machines (SVM), Random Forest, and Feedforward Neural Networks with explainable feature importance analysis.

Aligned with **UN Sustainable Development Goal 3 (Good Health and Well-being)**, specifically Targets 3.4 and 3.8, this project provides an accessible, interpretable, low-resource computational pipeline suitable for wearable screening and decentralized diagnostics.

---

## 📌 Key Highlights
- **Single-Lead ECG Analysis:** Minute-by-minute classification without cumbersome multi-channel polysomnography (PSG).
- **Comprehensive Preprocessing:** 4th-order Butterworth bandpass filtering (0.5–45 Hz), dual-window median baseline wander removal, and Pan-Tompkins QRS/R-peak detection.
- **Multi-Domain Feature Extraction (14 Features):**
  - **Time-Domain HRV (5):** Mean RR, SDNN, RMSSD, pNN50, Mean Heart Rate
  - **Frequency-Domain HRV (3):** LF power (0.04–0.15 Hz), HF power (0.15–0.40 Hz), LF/HF ratio via Welch PSD
  - **Wavelet Features (4):** Discrete Wavelet Transform (db4, Level 5), detail subband energy, variance, and Shannon wavelet entropy
  - **ECG-Derived Respiration (EDR) (2):** Amplitude modulation and respiration rate estimation
- **Comparative Classifiers:** Evaluated across SVM (RBF kernel), Random Forest (100 ensemble estimators), and Multilayer Perceptron (MLP Neural Network).
- **Explainable ML:** Random Forest Gini feature importances and ReliefF attribute ranking identifying critical apnea biomarkers.
- **Dual Pipeline Implementation:** Complete workflows implemented in both **Python** and **MATLAB**.
- **Interactive Browser Demo:** Standalone HTML/JS dashboard (`Project/demo/index.html`) for interactive record inspection and model score exploration.

---

## 📊 Performance Summary

Benchmark evaluation on the PhysioNet Apnea-ECG dataset:

| Model | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | AUC-ROC | Cohen's Kappa |
|---|---|---|---|---|---|---|---|
| **Random Forest** | 67.62% | **74.15%** | 65.07% | 45.42% | **0.5633** | **0.8209** | **0.3289** |
| **Neural Network (MLP)** | 70.31% | 60.54% | 74.13% | 47.85% | 0.5345 | 0.7414 | 0.3209 |
| **SVM (RBF Kernel)** | **73.37%** | 23.81% | **92.80%** | **56.45%** | 0.3349 | 0.7072 | 0.2015 |

> *Note: Random Forest demonstrates superior sensitivity (74.15%) and the highest AUC-ROC (0.8209), making it the most clinically suitable model for screening false-negative-sensitive apnea events.*

### Top Feature Importances (Random Forest)
1. `Var_D4` (Wavelet Detail D4 Variance) — **3.48**
2. `Wavelet_Entropy` (Shannon Wavelet Entropy) — **2.41**
3. `Ratio_D3_D4` (Subband Energy Ratio) — **2.18**
4. `Energy_A5` (Approximation A5 Energy) — **2.00**
5. `EDR_Amplitude` (ECG-Derived Respiration) — **1.37**

---

## 📁 Repository Structure

```text
APEX-Apnea-Prediction-from-ECG/
├── Project/
│   ├── code/
│   │   ├── 01_preprocessing/       # Bandpass filtering, Pan-Tompkins R-peak detector, baseline removal
│   │   ├── 02_features/            # Time, frequency, wavelet, and EDR feature extraction
│   │   ├── 03_classification/      # SVM, Random Forest, Feedforward NN training scripts
│   │   ├── 04_evaluation/          # Cross-validation, ablation study, statistical tests, external validation
│   │   ├── utils/                  # Dataset downloaders, class balance analysis, environment checker
│   │   ├── build_and_train.py      # End-to-end Python preprocessing, feature extraction & training pipeline
│   │   └── main_pipeline.m         # Master MATLAB execution script
│   ├── data/
│   │   └── apnea-ecg/              # PhysioNet Apnea-ECG dataset (.dat, .hea, .apn, .qrs)
│   ├── demo/
│   │   └── index.html              # Interactive client-side dashboard with model visualization
│   ├── paper/
│   │   └── data_notes.md           # Dataset analysis notes & validation logs
│   ├── results/
│   │   ├── figures/                # High-resolution ROC curves, confusion matrices
│   │   ├── models/                 # Serialized MATLAB model files (.mat)
│   │   ├── tables/                 # Performance tables, ablation study, ReliefF & RF ranking (.csv)
│   │   └── dashboard_data.json     # Precomputed evaluation payload for web demo
│   ├── detailed_project_plan.md    # Master project roadmap and experimental methodology
│   ├── project_schedule.md         # Phase-by-phase execution timeline
│   ├── Sleep_Apnea_ECG_SDG3_Research_Report.pdf # Comprehensive research report
│   └── README.md                   # MATLAB module documentation
├── results/                        # Root validation logs
├── .gitignore                      # Git exclusion rules
└── README.md                       # Main repository overview
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/OmkarSG18/APEX-Apnea-Prediction-from-ECG.git
cd APEX-Apnea-Prediction-from-ECG
```

### 2. Python Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install numpy scipy scikit-learn matplotlib pywavelets wfdb
```

### 3. Run the Python End-to-End Pipeline
```bash
cd Project
python code/build_and_train.py
```
This script will:
1. Load and parse the Apnea-ECG dataset records.
2. Filter the signals and detect R-peaks using the Pan-Tompkins method.
3. Compute the 14 multi-domain features per 60-second window.
4. Train SVM, Random Forest, and MLP models.
5. Export performance tables, ROC curves, confusion matrices, and web dashboard data.

### 4. Running the MATLAB Pipeline
1. Open MATLAB R2023b or R2024a.
2. Navigate to `Project/code/`.
3. Run the master pipeline:
   ```matlab
   main_pipeline
   ```

### 5. Launch the Interactive Demo
Open `Project/demo/index.html` in any modern web browser to interactively explore record features, ROC curves, and prediction results.

---

## 📈 Figures & Visualizations
Precomputed figures are stored under [`Project/results/figures/`](Project/results/figures/):
- **ROC Curves:** Comparison of SVM, Random Forest, and Neural Network receiver operating characteristics.
- **Confusion Matrix:** True vs. predicted apnea states under the best-performing Random Forest model.

---

## 🤝 Citation & References
If you find this codebase or methodology useful, please cite the primary dataset:
- Penzel, T., Moody, G. B., Mark, R. G., Goldberger, A. L., & Peter, J. H. (2000). *The Apnea-ECG Database*. Computers in Cardiology 2000, 27, 255-258. [PhysioNet](https://doi.org/10.13026/C2601R).
