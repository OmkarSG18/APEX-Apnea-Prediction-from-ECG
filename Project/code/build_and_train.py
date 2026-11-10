import os
import sys
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import scipy.interpolate as interp
import pywt
import wfdb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, cohen_kappa_score, roc_curve
import matplotlib.pyplot as plt
import csv

DATA_DIR = os.path.join('data', 'apnea-ecg')
RESULTS_DIR = 'results'
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

for d in [RESULTS_DIR, MODELS_DIR, TABLES_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

FS = 100
SEGMENT_LEN = 60 * FS # 6000 samples per minute

def bandpass_filter(sig, fs=100):
    b, a = signal.butter(4, [0.5, 45.0], btype='bandpass', fs=fs)
    return signal.filtfilt(b, a, sig)

def remove_baseline_median(sig, fs=100):
    w1 = int(0.200 * fs)
    w2 = int(0.600 * fs)
    if w1 % 2 == 0: w1 += 1
    if w2 % 2 == 0: w2 += 1
    temp = signal.medfilt(sig, w1)
    base = signal.medfilt(temp, w2)
    return sig - base

def pan_tompkins(sig, fs=100):
    b, a = signal.butter(3, [5.0, 15.0], btype='bandpass', fs=fs)
    filt = signal.filtfilt(b, a, sig)
    diff = np.diff(filt)
    sq = diff ** 2
    win = int(0.150 * fs)
    integrated = np.convolve(sq, np.ones(win)/win, mode='same')
    
    min_dist = int(0.600 * fs)
    height = 0.35 * np.max(integrated)
    peaks_int, _ = signal.find_peaks(integrated, distance=min_dist, height=height)
    
    r_locs = []
    search_w = int(0.100 * fs)
    N = len(sig)
    for p in peaks_int:
        st = max(0, p - search_w)
        en = min(N, p + search_w)
        if en > st:
            r_locs.append(st + np.argmax(sig[st:en]))
    
    r_locs = np.unique(r_locs)
    r_peaks = sig[r_locs]
    rr_raw = np.diff(r_locs) / fs
    
    rr_clean = rr_raw.copy()
    for i in range(2, len(rr_raw) - 2):
        med = np.median(rr_raw[i-2:i+3])
        if med > 0 and abs(rr_raw[i] - med) / med > 0.20:
            rr_clean[i] = np.nan
            
    return r_peaks, r_locs, rr_clean

def extract_time_hrv(rr_clean):
    rr = rr_clean[~np.isnan(rr_clean)]
    if len(rr) < 3:
        return [np.nan] * 5
    mRR = np.mean(rr)
    SDNN = np.std(rr)
    drr = np.diff(rr)
    RMSSD = np.sqrt(np.mean(drr**2)) if len(drr) > 0 else np.nan
    pNN50 = np.sum(np.abs(drr) > 0.050) / len(drr) if len(drr) > 0 else np.nan
    HR_mean = 60.0 / mRR if mRR > 0 else np.nan
    return [mRR, SDNN, RMSSD, pNN50, HR_mean]

def extract_freq_hrv(rr_clean, rr_times, fs_resample=4):
    valid = ~np.isnan(rr_clean)
    rr = rr_clean[valid]
    t = rr_times[valid]
    if len(rr) < 4 or (t[-1] - t[0]) < 10:
        return [np.nan, np.nan, np.nan]
    try:
        t_grid = np.arange(t[0], t[-1], 1.0/fs_resample)
        f_interp = interp.interp1d(t, rr, kind='cubic', fill_value='extrapolate')
        rr_uniform = f_interp(t_grid)
        rr_detrend = rr_uniform - np.mean(rr_uniform)
        
        freqs, psd = signal.welch(rr_detrend, fs=fs_resample, nperseg=min(len(rr_detrend), int(30*fs_resample)))
        
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
        
        LF = np.trapezoid(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0.0
        HF = np.trapezoid(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0.0
        LF_HF = LF / HF if HF > 0 else np.nan
        return [LF, HF, LF_HF]
    except:
        return [np.nan, np.nan, np.nan]

def extract_wavelet(ecg_seg):
    try:
        coeffs = pywt.wavedec(ecg_seg, 'db4', level=5)
        cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
        energies = [np.sum(c**2) for c in [cA5, cD5, cD4, cD3, cD2, cD1]]
        E_tot = np.sum(energies)
        prob = np.array(energies) / (E_tot + 1e-12)
        WE = -np.sum(prob * np.log2(prob + 1e-12))
        E_A5 = energies[0]
        Ratio_D3_D4 = energies[3] / (energies[2] + 1e-12)
        Var_D4 = np.var(cD4)
        return [WE, E_A5, Ratio_D3_D4, Var_D4]
    except:
        return [np.nan] * 4

def extract_edr(r_peaks, r_locs):
    if len(r_peaks) < 4:
        return [np.nan, np.nan]
    try:
        r_times = r_locs / FS
        t_grid = np.arange(r_times[0], r_times[-1], 0.25)
        f_interp = interp.interp1d(r_times, r_peaks, kind='cubic', fill_value='extrapolate')
        edr_sig = f_interp(t_grid)
        amp = np.max(edr_sig) - np.min(edr_sig)
        pks, _ = signal.find_peaks(edr_sig, distance=6)
        rate = len(pks)
        return [amp, rate]
    except:
        return [np.nan, np.nan]

def get_available_records():
    recs = []
    if not os.path.exists(DATA_DIR):
        return recs
    for fn in os.listdir(DATA_DIR):
        if fn.endswith('.dat'):
            r = fn[:-4]
            apn_p = os.path.join(DATA_DIR, r + '.apn')
            dat_p = os.path.join(DATA_DIR, r + '.dat')
            if os.path.exists(apn_p) and os.path.getsize(dat_p) > 1000 and os.path.getsize(apn_p) > 10:
                recs.append(r)
    return sorted(list(set(recs)))

def process_record_list(rec_list):
    X_list = []
    y_list = []
    for rec in rec_list:
        rec_path = os.path.join(DATA_DIR, rec)
        try:
            record = wfdb.rdrecord(rec_path)
            annotation = wfdb.rdann(rec_path, 'apn')
            sig = record.p_signal[:, 0]
            labels = annotation.symbol
            
            filt = bandpass_filter(sig)
            clean = remove_baseline_median(filt)
            r_peaks, r_locs, _ = pan_tompkins(clean)
            
            n_min = min(len(sig) // SEGMENT_LEN, len(labels))
            for m in range(n_min):
                st = m * SEGMENT_LEN
                en = (m + 1) * SEGMENT_LEN
                ecg_seg = clean[st:en]
                
                mask = (r_locs >= st) & (r_locs < en)
                m_peaks = r_peaks[mask]
                m_locs = r_locs[mask] - st
                m_rr = np.diff(m_locs) / FS
                m_rr_times = m_locs[1:] / FS
                
                t_f = extract_time_hrv(m_rr)
                f_f = extract_freq_hrv(m_rr, m_rr_times)
                w_f = extract_wavelet(ecg_seg)
                e_f = extract_edr(m_peaks, m_locs)
                
                row = t_f + f_f + w_f + e_f
                X_list.append(row)
                y_list.append(labels[m])
            print(f'Processed record {rec}: {n_min} segments')
        except Exception as e:
            print(f'Skipping record {rec}: {e}')
            
    return np.array(X_list), np.array(y_list)

def main():
    avail_recs = get_available_records()
    print(f'Found {len(avail_recs)} verified records in {DATA_DIR}: {avail_recs}')
    
    if len(avail_recs) == 0:
        print('No verified dataset records found yet.')
        return

    train_recs = [r for r in avail_recs if not r.startswith('x')]
    test_recs = [r for r in avail_recs if r.startswith('x')]

    print(f'Processing Training split ({len(train_recs)} records)...')
    X_tr_raw, y_tr_raw = process_record_list(train_recs)
    
    if len(test_recs) > 0:
        print(f'Processing Test split ({len(test_recs)} records)...')
        X_te_raw, y_te_raw = process_record_list(test_recs)
    else:
        X_te_raw, y_te_raw = np.array([]), np.array([])

    if len(X_te_raw) == 0:
        print('Splitting available training segments into 80% train / 20% test...')
        split_idx = int(0.8 * len(X_tr_raw))
        X_te_raw, y_te_raw = X_tr_raw[split_idx:], y_tr_raw[split_idx:]
        X_tr_raw, y_tr_raw = X_tr_raw[:split_idx], y_tr_raw[:split_idx]

    print(f'Final Train matrix shape: {X_tr_raw.shape}, Test matrix shape: {X_te_raw.shape}')

    # Impute NaNs using column medians from training split
    meds = np.nanmedian(X_tr_raw, axis=0)
    meds[np.isnan(meds)] = 0.0
    for col in range(X_tr_raw.shape[1]):
        X_tr_raw[np.isnan(X_tr_raw[:, col]), col] = meds[col]
        X_te_raw[np.isnan(X_te_raw[:, col]), col] = meds[col]
        
    # Standardize features using train statistics only
    mu = np.mean(X_tr_raw, axis=0)
    std = np.std(X_tr_raw, axis=0)
    std[std == 0] = 1.0
    
    X_train = (X_tr_raw - mu) / std
    X_test = (X_te_raw - mu) / std
    
    y_train = y_tr_raw
    y_test = y_te_raw
    
    feature_names = [
        'mRR', 'SDNN', 'RMSSD', 'pNN50', 'HR_mean',
        'LF_power', 'HF_power', 'LF_HF_ratio',
        'Wavelet_Entropy', 'Energy_A5', 'Ratio_D3_D4', 'Var_D4',
        'EDR_Amplitude', 'EDR_Rate'
    ]
    
    # Save MATLAB .mat files
    sio.savemat(os.path.join(RESULTS_DIR, 'feature_matrix_train.mat'), {
        'X_train': X_train,
        'y_train': y_train,
        'feature_names': feature_names
    })
    
    sio.savemat(os.path.join(RESULTS_DIR, 'feature_matrix_test.mat'), {
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names
    })
    
    print('Feature matrices saved to results/feature_matrix_train.mat and results/feature_matrix_test.mat!')
    
    # Train Classifiers
    print('\nTraining Support Vector Machine (SVM RBF)...')
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    print('Training Random Forest (100 Trees)...')
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
    rf.fit(X_train, y_train)
    
    print('Training Feedforward Neural Network [64, 32, 16]...')
    y_tr_num = (y_train == 'A').astype(int)
    y_te_num = (y_test == 'A').astype(int)
    
    nn = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=100, early_stopping=True, random_state=42)
    nn.fit(X_train, y_tr_num)
    
    # Save models & evaluate
    print('\nEvaluating Models on Benchmark Test Set...')
    perf_file = os.path.join(TABLES_DIR, 'model_performance.csv')
    with open(perf_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_Score', 'AUC', 'Kappa'])
        
        models = [('SVM (RBF)', svm, False), ('Random Forest', rf, False), ('Neural Network', nn, True)]
        for name, model, is_num in models:
            if is_num:
                preds_num = model.predict(X_test)
                preds = np.where(preds_num == 1, 'A', 'N')
                probs = model.predict_proba(X_test)[:, 1]
            else:
                preds = model.predict(X_test)
                probs = model.predict_proba(X_test)[:, list(model.classes_).index('A')]
            
            acc = accuracy_score(y_test, preds)
            sens = recall_score(y_test, preds, pos_label='A')
            spec = recall_score(y_test, preds, pos_label='N')
            prec = precision_score(y_test, preds, pos_label='A')
            f1 = f1_score(y_test, preds, pos_label='A')
            auc = roc_auc_score((y_test == 'A').astype(int), probs)
            kappa = cohen_kappa_score(y_test, preds)
            
            print(f'  {name:20s} -> Acc: {acc*100:.2f}%, Sens: {sens*100:.2f}%, Spec: {spec*100:.2f}%, F1: {f1*100:.2f}%, AUC: {auc:.4f}, Kappa: {kappa:.4f}')
            writer.writerow([name, f'{acc:.4f}', f'{sens:.4f}', f'{spec:.4f}', f'{prec:.4f}', f'{f1:.4f}', f'{auc:.4f}', f'{kappa:.4f}'])
            
    # Feature Importance
    imp_file = os.path.join(TABLES_DIR, 'rf_importance.csv')
    with open(imp_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'Importance_Score'])
        for fn, imp in zip(feature_names, rf.feature_importances_):
            writer.writerow([fn, f'{imp:.6f}'])
            
    # Feature Category Ablation Study
    ablation_file = os.path.join(TABLES_DIR, 'ablation_study.csv')
    subsets = [
        ('Time-Domain HRV', list(range(0, 5))),
        ('Frequency-Domain HRV', list(range(5, 8))),
        ('Wavelet Features', list(range(8, 12))),
        ('Hybrid (All Features)', list(range(0, 14)))
    ]
    
    with open(ablation_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature_Category', 'Num_Features', 'Accuracy', 'Sensitivity', 'Specificity', 'AUC'])
        for cat_name, cols in subsets:
            rf_sub = RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=42)
            rf_sub.fit(X_train[:, cols], y_train)
            p_sub = rf_sub.predict(X_test[:, cols])
            pr_sub = rf_sub.predict_proba(X_test[:, cols])[:, list(rf_sub.classes_).index('A')]
            
            acc = accuracy_score(y_test, p_sub)
            sens = recall_score(y_test, p_sub, pos_label='A')
            spec = recall_score(y_test, p_sub, pos_label='N')
            auc = roc_auc_score((y_test == 'A').astype(int), pr_sub)
            
            writer.writerow([cat_name, len(cols), f'{acc:.4f}', f'{sens:.4f}', f'{spec:.4f}', f'{auc:.4f}'])
            
    # Generate Publication Figures (300 DPI)
    fig, ax = plt.subplots(figsize=(7, 6))
    models_plot = [('SVM (RBF)', svm, False, 'blue'), ('Random Forest', rf, False, 'red'), ('Neural Network', nn, True, 'green')]
    for name, model, is_num, color in models_plot:
        if is_num:
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.predict_proba(X_test)[:, list(model.classes_).index('A')]
        fpr, tpr, _ = roc_curve((y_test == 'A').astype(int), probs)
        auc_v = roc_auc_score((y_test == 'A').astype(int), probs)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_v:.4f})', color=color, linewidth=2)
        
    ax.plot([0, 1], [0, 1], 'k--', label='Chance Baseline', linewidth=1)
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curve Comparison on PhysioNet Benchmark Test Set')
    ax.legend(loc='lower right')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curves.png'), dpi=300)
    plt.close()
    
    print('\nALL PIPELINE STAGES COMPLETED SUCCESSFULLY!')
    print(f'Results tables exported to: {TABLES_DIR}')
    print(f'Publication figures exported to: {FIGURES_DIR}')

if __name__ == '__main__':
    main()
