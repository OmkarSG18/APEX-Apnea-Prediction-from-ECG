# Detailed Master Plan — Sleep Apnea Detection from Single-Lead ECG
## Comparative Machine Learning Study · MATLAB R2023b/R2024a · SDG 3 Alignment
**Version:** 1.0
**Prepared:** 13 September 2026
**Planned start:** Monday, 14 September 2026
**Planned finish:** Friday, 9 October 2026 (4 weeks / 20 working days)
**Primary output:** Research paper + reproducible MATLAB codebase + results dashboard + publication-quality figures
---
## PART 1 — PROJECT DEFINITION
### 1.1 Research Question
Can single-lead ECG signals accurately detect sleep apnea on a minute-by-minute basis, and which feature extraction method (time-domain HRV vs. frequency-domain HRV vs. wavelet-based features) gives the best classification performance?
### 1.2 Hypothesis
A hybrid feature set combining time-domain HRV and wavelet-based features will outperform individual feature categories for apnea detection from single-lead ECG.
### 1.3 Objectives (from the approved research report)
1. Preprocess raw ECG signals to remove noise and detect R-peaks.
2. Extract three feature categories: time-domain HRV (5), frequency-domain HRV (3), wavelet-based (4), plus optional EDR (2).
3. Compare SVM, Random Forest, and feedforward Neural Network classifiers.
4. Evaluate with Accuracy, Sensitivity, Specificity, Precision, F1, AUC, Kappa.
5. Validate the best model on a secondary dataset (MIT-BIH PSG or UCDDB).
### 1.4 Success Criteria (quantitative targets)
| Metric | Target |
|---|---|
| Accuracy | > 85% |
| Sensitivity (recall) | > 85% (critical for medical screening) |
| Specificity | > 85% |
| Precision | > 80% |
| F1-score | > 82% |
| AUC-ROC | > 0.90 |
| Kappa | > 0.70 |
### 1.5 Datasets (all open-access via PhysioNet)
| Dataset | Records | Sampling | Role |
|---|---|---|---|
| PhysioNet Apnea-ECG (primary) | 70 overnight recordings, 100 Hz, ~8–10 h each; ~34,347 minute-level labeled segments (17,045 train / 17,302 test); pre-defined 35/35 split | 100 Hz | Training + benchmark testing |
| MIT-BIH Polysomnographic (slpdb) | 18 recordings, 80+ hours total, ECG at 250 Hz | 250 Hz | Cross-validation / feature validation |
| UCD Sleep Apnea (ucddb) | 25 full polysomomgrams, 3-channel Holter ECG at 128 Hz, AASM event annotations | 128 Hz | External validation (generalizability) |
### 1.6 Required MATLAB Toolboxes
- Signal Processing Toolbox (`designfilt`, `filtfilt`, `findpeaks`, `pwelch`)
- Wavelet Toolbox (`wavedec`, `appcoef`, `detcoef`, `wentropy`, `wdenoise`)
- Statistics and Machine Learning Toolbox (`fitcsvm`, `TreeBagger`, `fitcnet`, `relieff`, `confusionmat`, `perfcurve`, `crossval`, `smote`)
- Curve Fitting Toolbox (spline interpolation support) — optional but recommended
- Deep Learning Toolbox — optional (only if the NN is later extended)
---
## PART 2 — WORKSPACE & FILE STRUCTURE
### Step 1 — Create the project folder layout (do this first)
Create exactly this structure before writing any code:
```
SleepApnea_ECG/
├── data/
│   ├── apnea-ecg/          % PhysioNet Apnea-ECG: .dat .hea .apn .qrs
│   ├── slpdb/              % MIT-BIH PSG
│   └── ucddb/              % UCD Sleep Apnea
├── code/
│   ├── 01_preprocessing/
│   ├── 02_features/
│   ├── 03_classification/
│   ├── 04_evaluation/
│   └── utils/
├── results/
│   ├── figures/            % 300 DPI PNG/EPS
│   ├── tables/             % exported performance tables (.csv/.mat)
│   └── models/             % trained model .mat files
├── paper/
└── README.md
```
**Verification for Step 1:** All folders exist on disk; `README.md` is created with a one-paragraph project description and the folder map above.
---
## PART 3 — PHASE 0: ENVIRONMENT SETUP & DATA ACQUISITION (Days 1–2)
### Step 2 — Verify MATLAB installation and toolboxes (Day 1, ~1 hour)
1. Launch MATLAB R2023b or R2024a.
2. Run `ver` and confirm these toolboxes are listed: Signal Processing, Wavelet, Statistics and Machine Learning.
3. If any toolbox is missing: open the Add-On Explorer, install it, or switch to MATLAB Online (free student license) if the local license does not cover it.
4. Record the exact MATLAB version and toolbox list in `README.md` for reproducibility.
**Verification:** `ver` output saved to `results/setup_verification.txt` (use `diary` to capture it).
### Step 3 — Install the WFDB Toolbox for MATLAB (Day 1, ~30 minutes)
1. Download from https://physionet.org/content/wfdb-matlab/ (or use `wfdb-installer`).
2. Add it to the MATLAB path: `addpath(genpath('path/to/wfdb-matlab'))`, then `savepath`.
3. Test: `[signal, fs, tm] = rdsamp('a01', 'data/apnea-ecg');` — confirm it returns a signal of length ≈ fs × (8–10 h) at fs = 100.
4. Test annotation reading: `[ann, type] = rdann('a01', 'apn', 'data/apnea-ecg');` — confirm annotation type 'A'/'N' labels are returned.
**Verification:** Both test commands run without error; sampling rate is 100 Hz.
### Step 4 — Download all three datasets (Day 1, remainder)
1. Download Apnea-ECG from https://physionet.org/content/apnea-ecg/1.0.0/ into `data/apnea-ecg/`. Files: 70 records, each with `.dat`, `.hea`, `.apn`; some with `.qrs`.
2. Download MIT-BIH PSG from https://physionet.org/content/slpdb/1.0.0/ into `data/slpdb/`.
3. Download UCDDB from https://physionet.org/content/ucddb/1.0.0/ into `data/ucddb/`.
4. Do not rename, move, or modify any downloaded file (originals are read-only inputs).
**Verification for Step 4:** File count check — Apnea-ECG directory contains all 70 records' files; write a small script `utils/check_dataset_files.m` that lists records and confirms each has .dat, .hea, and .apn present.
### Step 5 — Data exploration and understanding note (Day 2)
1. For one training record (e.g., `a01`): plot 60 seconds of raw ECG; note noise, baseline drift, and amplitude scale.
2. Load the `.apn` annotations for `a01`; count the number of 'A' vs 'N' minutes; compute the class balance.
3. Repeat the class-balance count across all 70 records and both splits (35 train / 35 test); store the counts in `results/tables/class_balance.csv`.
4. Confirm the sampling rate is uniformly 100 Hz for all records (read each `.hea` header).
5. Write a half-page data understanding note in `paper/data_notes.md`: record durations, label distribution, any artifacts observed, and the train/test record lists.
**Verification for Step 5:** `class_balance.csv` exists; note confirms ~34,347 total minute segments and a near-50/50 class distribution (flag if not — SMOTE may be needed later).
**Milestone M1 (end of Day 2):** Datasets acquired and verified, MATLAB environment and WFDB Toolbox working, data format fully documented.
---
## PART 4 — PHASE 1: PREPROCESSING & R-PEAK DETECTION (Days 3–5)
### Step 6 — Implement bandpass filtering (Day 3)
1. Write `code/01_preprocessing/bandpass_filter.m` implementing a 4th-order IIR bandpass, 0.5–45 Hz, at fs = 100:
   ```
   d = designfilt('bandpassiir', 'FilterOrder', 4, ...
       'HalfPowerFrequency1', 0.5, 'HalfPowerFrequency2', 45, ...
       'SampleRate', 100);
   filtered = filtfilt(d, ecg_signal);   % zero-phase: no temporal distortion
   ```
2. Apply to the full Apnea-ECG training set; save filtered signals (or filter on demand within the pipeline — choose one strategy and document it in `README.md`).
3. Visual check: plot 30 s of raw vs. filtered signal for 3 records with visibly different noise levels; confirm muscle-artifact noise is attenuated and QRS morphology is preserved.
**Verification for Step 6:** Filtered signal for a known record shows flat baseline outside QRS complexes and no phase shift of R-peaks versus raw signal.
### Step 7 — Baseline wander removal (Day 4)
1. Implement Method A in `code/01_preprocessing/remove_baseline_median.m`: subtract a median filter with a 200 ms window, then subtract a second median filter with a 600 ms window (standard two-stage ECG baseline removal).
2. Implement Method B in `code/01_preprocessing/remove_baseline_wavelet.m`: `wdenoise(signal, 8, 'Wavelet', 'db6')`, remove the coarsest approximation coefficients, reconstruct.
3. Run both on the same 3 test records; compare visually. Select the method with cleaner ST-segment stability and document the choice + rationale in `paper/data_notes.md`.
4. Set the chosen method as the single pipeline default — do not keep both branches active downstream.
**Verification for Step 7:** A 10-second plot of the output shows no residual slow drift; the QRS amplitude range is consistent across the night.
### Step 8 — Pan-Tompkins R-peak detection (Day 5)
1. Write `code/01_preprocessing/pan_tompkins.m` as a function `[peaks, locs] = pan_tompkins(ecg, fs)`:
   - Bandpass stage (5–15 Hz) for the detector: `butter(3, [5 15]/(fs/2), 'bandpass')` + `filtfilt`.
   - Differentiation: `d = diff(filtered);`
   - Squaring: `s = d.^2;`
   - Moving-window integration over 150 ms: `m = movmean(s, round(0.15*fs));`
   - Peak picking: `findpeaks(m, 'MinPeakDistance', round(0.6*fs), 'MinPeakHeight', 0.5*max(m));`
   - Map integrated-signal peaks back to the nearest local maximum in the filtered ECG for accurate R-peak timing.
2. Adaptive threshold: if any inter-beat gap exceeds 2 s, lower the peak-height threshold locally and re-scan that window (handles rate variation across the night).
3. Compute RR intervals: `rr = diff(locs) / fs;` and RR timestamps.
4. Ectopic-beat cleaning: flag RR intervals deviating > 20% from a 5-beat local median; exclude flagged beats from downstream HRV features (do not delete the beats themselves from the record).
**Verification for Step 8 (mandatory):** For every record that has PhysioNet `.qrs` annotations, compute detection agreement; target ≥ 99% beat match tolerance ±1 sample. Manually inspect 5–10 segments where disagreement occurs and record causes (noise bursts, missed low-amplitude beats). Log per-record detection accuracy to `results/tables/rpeak_validation.csv`.
**Milestone M2 (end of Day 5):** Preprocessing pipeline complete and validated; clean RR interval time series exist for all 70 recordings.
**Buffer rule:** If R-peak validation falls below target, use the morning of Day 6 for threshold tuning. Do not proceed to feature extraction with unvalidated R-peaks.
---
## PART 5 — PHASE 2: FEATURE EXTRACTION (Days 6–10)
All features are computed per minute, aligned exactly with the PhysioNet A/N minute boundaries from the `.apn` annotations.
### Step 9 — Time-domain HRV features (Day 6)
Write `code/02_features/extract_time_hrv.m`. For each minute window of NN (cleaned RR) intervals compute:
1. mRR = `mean(rr)` (mean RR interval, seconds)
2. SDNN = `std(rr)` (total variability)
3. RMSSD = `sqrt(mean(diff(rr).^2))` (short-term variability)
4. pNN50 = `sum(abs(diff(rr)) > 0.05) / length(rr)` (parasympathetic marker)
5. HR_mean = `60 / mean(rr)` (mean heart rate, bpm)
Output: `results/tables/features_time.csv` (one row per minute segment, columns = 5 features + record ID + minute index + label).
**Verification:** Spot-check 3 minutes by hand against a calculator; confirm units are seconds (RR) and bpm (HR); confirm each row's label matches the `.apn` annotation for that minute.
### Step 10 — Frequency-domain HRV features (Day 7)
Write `code/02_features/extract_freq_hrv.m`:
1. Within each minute, take the cleaned RR intervals and their timestamps.
2. Resample the uneven RR series to a uniform 4 Hz grid using cubic spline: `interp1(rr_times, rr, new_times, 'cubic')` where `new_times` is the 4 Hz grid spanning the minute.
3. Compute the power spectral density with `pwelch` (4 Hz resampled signal; use a window that fits the ~60 s segment).
4. Integrate power in bands: LF = 0.04–0.15 Hz (`bandpower` or `pwelch` over `[0.04:0.001:0.15]`); HF = 0.15–0.4 Hz.
5. LF/HF ratio = LF / HF.
Output: `results/tables/features_freq.csv` (3 features per minute).
**Verification for Step 10:** If a minute contains fewer than 3 valid RR intervals, mark LF/HF/LFHF as NaN (handle in Step 13). Confirm no spectral leakage artifacts: plot one example minute's PSD and verify band boundaries.
### Step 11 — Wavelet-based features (Day 8)
Write `code/02_features/extract_wavelet.m` taking a 1-minute raw ECG segment (preprocessed):
1. ` [c, l] = wavedec(segment, 5, 'db4');`
2. Extract: cA5 = `appcoef(c, l, 'db4', 5)`; cD5, cD4, cD3, cD2, cD1 = `detcoef(c, l, level)`.
3. Feature 1 — Wavelet entropy (WE): energies of the 6 coefficient sets; `prob = energies/sum(energies)`; `WE = -sum(prob .* log2(prob + eps));`
4. Feature 2 — Approximation energy A5: `sum(cA5.^2)`
5. Feature 3 — Detail energy ratio D3/D4: `sum(cD3.^2) / sum(cD4.^2)`
6. Feature 4 — Wavelet variance: `var(cD4)`
Output: `results/tables/features_wavelet.csv` (4 features per minute).
**Verification for Step 11:** Run on 3 apnea minutes and 3 normal minutes; confirm the wavelet entropy visibly differs between the two classes (it should be higher during apnea — irregular ECG); if not, recheck coefficient indexing.
### Step 12 — ECG-derived respiration (EDR) features (Day 9)
Write `code/02_features/extract_edr.m`:
1. Take R-peak amplitudes from the preprocessed signal at the validated R-peak locations.
2. Interpolate amplitudes to a uniform grid (e.g., 4 Hz) — this is the EDR amplitude signal.
3. Feature 1 — EDR amplitude: peak-to-peak of the EDR signal within the minute.
4. Feature 2 — EDR rate: `findpeaks` on the interpolated EDR signal; breathing rate = number of peaks per minute.
Output: `results/tables/features_edr.csv` (2 features per minute).
**Verification for Step 12:** Plot EDR amplitude vs. time alongside the label sequence for one record; apnea stretches should show reduced/modulated breathing pattern. If EDR is too noisy on 100 Hz single-lead data, document it and continue — EDR is optional, the 12 HRV/wavelet features are the core.
### Step 13 — Assemble the final feature matrix (Day 10)
1. Write `code/02_features/build_feature_matrix.m`: join all per-minute feature tables on (record ID, minute index); attach the label (A/N) from the `.apn` annotations.
2. Handle NaNs: minutes with < 3 valid RR intervals → median-impute from the same record's neighboring minutes; if an entire record fails quality, exclude it and document the exclusion.
3. Standardize: z-score each feature using training-split statistics only — `mu = mean(X_train); sigma = std(X_train); X = (X - mu) ./ sigma;` Apply the same mu/sigma to the test split (no data leakage).
4. Correlation pruning: compute the feature correlation matrix on the training split; if any pair has |r| > 0.9, drop the lower-ranked feature (rank by univariate AUC) and record the decision.
5. Produce the final matrix: expected size ≈ 34,347 rows × 12–14 columns; save as `results/feature_matrix_train.mat` and `results/feature_matrix_test.mat`.
6. Visualization: class-wise distribution plots (histogram/boxplot per feature), correlation heatmap → save to `results/figures/`.
**Verification for Step 13:** Row counts match the segment counts from Step 5 (17,045 train / 17,302 test, minus documented exclusions); no NaN remains; train/test standardization used identical mu/sigma.
**Milestone M3 (end of Day 10):** Final standardized feature matrices ready for classification, with documented quality-control decisions.
---
## PART 6 — PHASE 3: CLASSIFICATION MODEL TRAINING (Days 11–13)
All training uses the pre-defined 35-record training split only. The test split remains untouched until Phase 4 evaluation.
### Step 14 — SVM training (Day 11)
Write `code/03_classification/train_svm.m`:
1. Inputs: `feature_matrix_train.mat` (X_train, y_train as categorical 'A'/'N').
2. Train: `fitcsvm(X_train, y_train, 'KernelFunction', 'rbf', 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', 'MaxObjectiveEvaluations', 30, 'Kfold', 5));`
3. If class imbalance was flagged in Step 5, set `'ClassNames'` weighting or oversample with `smote` beforehand.
4. Save model: `results/models/svm_model.mat`; record chosen BoxConstraint and KernelScale in a run log.
**Verification:** Training completes; 5-fold CV estimate of accuracy is in a plausible range (78–88% per literature expectations); no convergence warnings.
### Step 15 — Random Forest training (Day 12)
Write `code/03_classification/train_rf.m`:
1. Train: `rf = TreeBagger(100, X_train, y_train, 'Method', 'classification', 'OOBPrediction', 'on', 'MinLeafSize', 5);`
2. Plot OOB error vs. number of trees; if error is still decreasing at 100 trees, retrain with up to 300 trees and record the final choice.
3. Extract predictor importance (`rf.OOBPermutedPredictorDeltaError`) and save to `results/tables/rf_importance.csv`.
4. Save model: `results/models/rf_model.mat`.
**Verification:** OOB accuracy is stable; importance table has one entry per feature.
### Step 16 — Neural Network training (Day 13)
Write `code/03_classification/train_nn.m`:
1. Hold out 15% of the training split as an internal validation set for early stopping (stratified by label).
2. Train: `fitcnet(X_train, y_train, 'LayerSizes', [64 32 16], 'Activations', 'relu', 'MaxEpochs', 100, 'ValidationData', {X_val, y_val}, 'ValidationPatience', 10);`
3. Record training/validation loss curves; confirm early stopping triggered (or that 100 epochs completed without divergence).
4. Save model: `results/models/nn_model.mat` plus the loss-curve figure.
**Verification:** Validation loss does not diverge from training loss (no severe overfitting); final validation accuracy is plausible.
**Milestone M4 (end of Day 13):** Three classifiers trained on the hybrid feature set; hyperparameters and models saved.
---
## PART 7 — PHASE 4: EVALUATION, VALIDATION & VISUALIZATION (Days 14–17)
### Step 17 — Cross-validation and robustness testing (Day 14)
Write `code/04_evaluation/run_crossval.m`:
1. Wrap each model with `crossval(mdl, 'KFold', 10)` on the training split; report `kfoldLoss`, `kfoldPredict`.
2. Compute per-fold accuracy; report mean ± standard deviation; save to `results/tables/cv_results.csv`.
3. Apply each trained model to the untouched test split; compute full metric set per model: Accuracy, Sensitivity, Specificity, Precision, F1, AUC (`perfcurve`), Kappa (`confusionmat`-derived or custom).
4. If class imbalance was flagged: apply `smote` to the training split and retrain the best model; compare and document.
**Verification for Step 17:** Test metrics are computed strictly from predictions on test records never used in training or tuning; all numbers flow from saved prediction vectors, not re-estimated ad hoc.
### Step 18 — Feature importance and ablation (Day 15)
Write `code/04_evaluation/feature_analysis.m`:
1. ReliefF ranking: `relieff(X_train, y_train, 10)`; save ranks to `results/tables/relieff_ranking.csv`.
2. Combine with RF importance from Step 15 into a single ranked table.
3. Feature-category ablation: retrain the best classifier on each subset — time-only, frequency-only, wavelet-only, hybrid (time+freq+wavelet) — using identical hyperparameters; record the 4 accuracy/AUC values. This table directly answers the research question.
4. Optional: retrain best model on ReliefF top-k features (k = 8, 10) to test whether selection improves performance.
**Verification:** The ablation table has exactly 4 rows with consistent metrics; the conclusion about which category contributes most is supported by at least the ablation + importance rankings.
### Step 19 — Statistical comparison and figures (Day 16)
Write `code/04_evaluation/statistical_tests.m` and `code/04_evaluation/make_figures.m`:
1. McNemar's test between each pair of classifiers on the test split predictions (custom implementation using the discordant counts b, c: chi-square = (b−c)²/(b+c)); save p-values to `results/tables/mcnemar.csv`.
2. Generate publication-quality figures (300 DPI, `exportgraphics`, consistent fonts/sizing):
   - ROC curves for all three classifiers on one plot (with AUC in legend).
   - Confusion matrices (normalized) for all three classifiers.
   - Precision-recall curves.
   - Feature correlation heatmap.
   - Class-wise boxplots for the top-4 features.
   - Bar chart of ablation results (feature categories vs. accuracy).
3. Save all figures to `results/figures/` in both PNG and EPS.
**Verification:** Every figure renders without clipping; axes labeled with units; numbers in figures match `results/tables/` values exactly.
### Step 20 — External validation (Day 17)
Write `code/04_evaluation/external_validation.m`:
1. Choose one validation dataset (MIT-BIH PSG preferred; UCDDB as alternative) — decision recorded in the run log.
2. Resample the validation ECG to 100 Hz (or run the pipeline natively at its own fs by parameterizing all functions on `fs` — verify each function accepts fs as an argument).
3. Run the identical preprocessing → feature extraction pipeline on the validation recordings; map its event annotations to minute-level A/N labels using the same rule (any apnea event overlapping a minute → 'A').
4. Apply the best trained model (no retraining); report metrics to `results/tables/external_validation.csv`.
5. Document any domain shift: different patient population, recording conditions, resampling effects.
**Verification for Step 20:** Metrics computed with the same code path as the test evaluation; any preprocessing deviation from the main pipeline is explicitly documented.
**Buffer rule:** If Days 14–16 overran, Step 20 may be shortened to 5–10 validation recordings instead of the full set, or dropped as the designated contingency task — the core comparative results must not be compressed.
**Milestone M5 (end of Day 17):** Complete comparative evaluation (feature categories × classifiers), statistical significance established, external validation results documented.
---
## PART 8 — PHASE 5: PAPER WRITING & FINAL DOCUMENTATION (Days 18–20)
### Step 21 — Write Introduction and Related Work (Day 18)
1. Introduction (~1,000 words): sleep apnea definition, prevalence (936 million affected; 80% undiagnosed), health consequences (hypertension, heart disease, stroke, diabetes), PSG cost/accessibility gap, ECG physiological rationale (autonomic dysfunction during apnea), SDG 3.4/3.8 framing, research gap (systematic classical-ML comparison with interpretability for resource-constrained settings), objectives.
2. Related Work (~800 words): review 5–7 key papers — de Chazal et al. 2000 (82.2% baseline), Sharma et al. 2017 (wavelet+SVM, 87.5%), Biswas et al. 2019 (1D-CNN, 91.2%), Zhang et al. 2022 (multi-scale entropy + LSTM, 93.1%), Mendez et al. 2014 (EDR review), Khandoker et al. 2020 (HRV review); position this work as the interpretable, reproducible, low-resource alternative to deep-learning trend.
**Verification:** Every claim carries a citation; reference manager (Zotero) library has the full list.
### Step 22 — Write Materials & Methods and Results (Day 19)
1. Materials & Methods (~1,500 words): dataset descriptions (tables from the report), preprocessing with all parameter values (filter order, cutoffs, wavelet, thresholds), mathematical definitions of all 12+ features with equations, classifier specifications and hyperparameter ranges, evaluation protocol (splits, CV, metrics, McNemar).
2. Results (~1,000 words): feature importance table, classifier comparison table, ablation table, external validation table, ROC/confusion-matrix/PR figures, statistical test results. Every number copied verbatim from `results/tables/` files — no hand re-typing of recomputed values.
**Verification for Step 22:** Cross-check each table number against the corresponding CSV; equation symbols match the text.
### Step 23 — Discussion, Conclusion, Abstract & final assembly (Day 20)
1. Discussion (~800 words): interpret why hybrid features outperform single categories (complementary information: autonomic balance + morphological variability), clinical implications for wearable screening, limitations (
