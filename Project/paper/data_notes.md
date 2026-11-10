# Data Understanding & Exploratory Notes

## Primary Dataset: PhysioNet Apnea-ECG Database

- **Source:** PhysioNet (https://physionet.org/content/apnea-ecg/1.0.0/)
- **Records:** 70 overnight ECG recordings (35 training, 35 testing).
- **Sampling Frequency ($f_s$):** 100 Hz (1 sample every 10 ms).
- **Duration:** Approximately 7 to 10 hours per recording.
- **Annotations:**
  - `.apn`: Minute-by-minute expert sleep apnea labels ('A' = Apnea, 'N' = Normal).
  - `.qrs`: Reference R-peak detection markers for validation.
  - `.hea`: Header file containing signal format, gain, and channel details.
  - `.dat`: Binary raw single-lead ECG signal.

---

## Data Structure & Segment Breakdown
- Total Minute-Level Labeled Segments: ~34,347 segments across 70 recordings.
  - Training Split (35 records): ~17,045 minute segments.
  - Benchmark Test Split (35 records): ~17,302 minute segments.
- Segment Alignment: Each minute segment (60 seconds = 6,000 samples @ 100 Hz) aligns precisely with one annotation in `.apn`.

---

## Signal Artifacts & Quality Observations
1. **Baseline Drift:** Low-frequency respiration and patient body movement induce baseline wander (< 0.5 Hz).
2. **High-Frequency Noise:** Electromyographic (EMG) muscle artifacts and 50/60 Hz powerline interference.
3. **Ectopic Beats:** Premature ventricular contractions (PVCs) and premature atrial contractions (PACs) present in certain recordings, requiring RR interval median filtering prior to HRV extraction.
