# Project Schedule — Sleep Apnea Detection from Single-Lead ECG

**Project type:** Comparative machine learning study
**Platform:** MATLAB R2023b / R2024a
**Primary dataset:** PhysioNet Apnea-ECG Database (70 recordings, 100 Hz, minute-level labels)
**Secondary validation dataset:** MIT-BIH Polysomnographic Database / UCD Sleep Apnea Database
**SDG alignment:** Goal 3 — Good Health and Well-being (Targets 3.4, 3.8)
**Total duration:** 4 weeks (20 working days, Mon–Fri), starting Monday, 14 September 2026
**Final deliverables:** Research paper + documented MATLAB code + results dashboard + publication-quality figures

---

## Phase Overview

| Phase | Focus | Weeks | Working Days |
|---|---|---|---|
| 0 | Environment Setup & Data Acquisition | Week 1 (Days 1–2) | 2 |
| 1 | Signal Preprocessing & R-Peak Detection | Week 1 (Days 3–5) | 3 |
| 2 | Feature Extraction | Week 2 (Days 6–10) | 5 |
| 3 | Classification Model Training | Week 3 (Days 11–13) | 3 |
| 4 | Evaluation, Validation & Visualization | Week 3–4 (Days 14–17) | 4 |
| 5 | Paper Writing & Final Documentation | Week 4 (Days 18–20) | 3 |

---

## Phase 0 — Environment Setup & Data Acquisition (Days 1–2)

| Day | Date | Task | Deliverable | Duration |
|---|---|---|---|---|
| 1 | Mon 14 Sep | Install/verify MATLAB R2023b–R2024a with Signal Processing, Wavelet, and Statistics & Machine Learning Toolboxes; install WFDB Toolbox for MATLAB | Working environment | 1 day |
| 1 | Mon 14 Sep | Download PhysioNet Apnea-ECG dataset (70 recordings: .dat, .hea, .apn, .qrs files); download MIT-BIH PSG and UCDDB for later validation | Dataset loaded in MATLAB | 1 day (parallel) |
| 2 | Tue 15 Sep | Explore data structure: sampling rate (100 Hz), record durations (~8–10 h), annotation format (A/N per minute); verify train/test split (35 + 35 recordings, ~34,347 labeled minute segments) | Data understanding note | 1 day |

**Milestone M1 — End of Day 2:** Datasets acquired, toolboxes verified, data format fully understood.

---

## Phase 1 — Signal Preprocessing & R-Peak Detection (Days 3–5)

| Day | Date | Task | Deliverable | Duration |
|---|---|---|---|---|
| 3 | Wed 16 Sep | Implement bandpass filter (0.5–45 Hz, 4th-order IIR, zero-phase `filtfilt`) to remove muscle-artifact noise and baseline wander | Filtered ECG signals | 1 day |
| 4 | Thu 17 Sep | Implement baseline wander removal (median filter 200 ms/600 ms windows; DWT-based `wdenoise` with db6 as cross-check) | Clean ECG signals | 1 day |
| 5 | Fri 18 Sep | Implement Pan-Tompkins R-peak detection (differentiation → squaring → moving-window integration → adaptive threshold); validate against PhysioNet .qrs annotations; compute and clean RR intervals (remove ectopic beats) | Validated R-peak locations + RR interval time-series for all 70 recordings | 1 day |

**Milestone M2 — End of Day 5:** Preprocessing pipeline complete; R-peak detection validated against reference annotations; clean RR series ready for feature extraction.

**Risk buffer:** R-peak detection errors → adaptive thresholding + manual inspection of 5–10 segments; absorb slip into Day 6 morning if needed.

---

## Phase 2 — Feature Extraction (Days 6–10)

| Day | Date | Task | Deliverable | Duration |
|---|---|---|---|---|
| 6 | Mon 21 Sep | Extract time-domain HRV features per minute: mRR, SDNN, RMSSD, pNN50, mean HR (5 features) | Feature matrix: time-domain | 1 day |
| 7 | Tue 22 Sep | Resample RR intervals to 4 Hz (cubic spline); compute frequency-domain features: LF power (0.04–0.15 Hz), HF power (0.15–0.4 Hz), LF/HF ratio (3 features) | Feature matrix: frequency-domain | 1 day |
| 8 | Wed 23 Sep | Implement DWT decomposition (db4, level 5); extract wavelet entropy, approximation energy A5, detail-energy ratio D3/D4, wavelet variance (4 features) | Feature matrix: wavelet | 1 day |
| 9 | Thu 24 Sep | Extract ECG-derived respiration (EDR) features: R-peak amplitude modulation + breathing rate | Feature matrix: EDR | 1 day |
| 10 | Fri 25 Sep | Combine all features into final matrix (~N × 12+); z-score normalization; correlation analysis (remove features with r > 0.9); plot class-wise feature distributions | Final standardized feature matrix + feature analysis plots | 1 day |

**Milestone M3 — End of Day 10:** Complete, normalized feature matrix ready for classification; feature correlation and distribution analysis documented.

---

## Phase 3 — Classification Model Training (Days 11–13)

| Day | Date | Task | Deliverable | Duration |
|---|---|---|---|---|
| 11 | Mon 28 Sep | Train SVM (RBF kernel) with Bayesian hyperparameter optimization (BoxConstraint, KernelScale); respect pre-defined train/test split | Trained SVM + metrics | 1 day |
| 12 | Tue 29 Sep | Train Random Forest (100 trees, `sqrt` features per split, OOB error monitoring); tune tree count if needed | Trained RF + metrics | 1 day |
| 13 | Wed 30 Sep | Train feedforward neural network (hidden layers [64, 32, 16], ReLU, Adam, 100 epochs, early stopping, validation split) | Trained NN + metrics | 1 day |

**Milestone M4 — End of Day 13:** All three classifiers trained on the full hybrid feature set with baseline metrics (accuracy, sensitivity, specificity, precision, F1, AUC).

---

## Phase 4 — Evaluation, Validation & Visualization (Days 14–17)

| Day | Date | Task | Deliverable | Duration |
|---|---|---|---|---|
| 14 | Thu 1 Oct | Run 10-fold cross-validation for all three models; compute confidence intervals; apply SMOTE or class weighting if imbalance found | Cross-validated performance report | 1 day |
| 15 | Fri 2 Oct | Feature importance analysis: ReliefF ranking + Random Forest Gini importance; optional feature-selection re-run of best model | Feature ranking table | 1 day |
| 16 | Mon 5 Oct | Statistical comparison: McNemar's test between classifiers; generate ROC curves, confusion matrices, precision-recall curves (300 DPI, publication quality) | Statistical report + evaluation figure set | 1 day |
| 17 | Tue 6 Oct | External validation of the best model on MIT-BIH PSG (18 recordings, 250 Hz ECG) or UCDDB (25 recordings, 128 Hz); document any domain-shift handling (resampling, re-labeling) | External validation results | 1 day |

**Milestone M5 — End of Day 17:** Full comparative evaluation complete — feature categories × classifiers matrix, statistical significance established, external validation confirming generalizability. Target thresholds: accuracy > 85%, sensitivity > 85%, AUC > 0.90.

**Buffer note:** External validation (Day 17) is the designated drop-task if slippage occurs; core comparison results stand without it.

---

## Phase 5 — Paper Writing & Final Documentation (Days 18–20)

| Day | Date | Task | Deliverable | Duration |
|---|---|---|---|---|
| 18 | Wed 7 Oct | Write Introduction (~1,000 words: prevalence, PSG access gap, ECG rationale, SDG 3 framing, research gap) + Related Work (~800 words: 5–7 key papers from de Chazal 2000 to Zhang 2022) | Draft: Sections 1–2 | 1 day |
| 19 | Thu 8 Oct | Write Materials & Methods (~1,500 words: datasets, preprocessing parameters, feature equations, classifier hyperparameters, evaluation protocol) + Results (~1,000 words with all tables/figures) | Draft: Sections 3–4 | 1 day |
| 20 | Fri 9 Oct | Write Discussion (~800 words: hybrid-feature interpretation, clinical/wearable implications, limitations) + Conclusion (~300 words) + Abstract (~250 words); format references (IEEE or APA, 20–30 citations); final code documentation and README; assemble results dashboard | Final paper + clean documented code | 1 day |

**Milestone M6 — End of Day 20:** Project complete — final research paper, reproducible MATLAB codebase, results dashboard, and publication-quality figure set delivered.

---

## Key Milestones Summary

| Milestone | Date | Definition of Done |
|---|---|---|
| M1 | Tue 15 Sep | Datasets acquired and understood; MATLAB environment verified |
| M2 | Fri 18 Sep | Validated preprocessing pipeline; RR intervals ready |
| M3 | Fri 25 Sep | Standardized feature matrix complete |
| M4 | Wed 30 Sep | Three classifiers trained on hybrid features |
| M5 | Tue 6 Oct | Comparative evaluation + external validation complete |
| M6 | Fri 9 Oct | Final paper + code + dashboard delivered |

---

## Post-Project Path (Outside the 4-Week Schedule)

| Activity | Timeline | Notes |
|---|---|---|
| ArXiv preprint + public GitHub repo | Immediate | Fast visibility, no peer review |
| MDPI Sensors / Bioengineering submission | +2–4 months | Open access, fast review |
| IEEE conference submission (EMBC / BIBE / ICASSP biomedical track) | +4–6 months | Good early-stage venue |

---

## Dependencies & Critical Path

Dataset download → Preprocessing → Feature extraction → Classification → Evaluation → Paper writing. Preprocessing (Phase 1) is the highest-risk phase: all downstream work depends on validated R-peaks, so the Day 6 morning buffer exists to absorb detection-tuning slippage without compressing feature extraction. Classification days are parallelizable (SVM/RF/NN are independent) if compute time becomes a constraint on an older machine.
