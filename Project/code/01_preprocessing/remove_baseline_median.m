function clean_ecg = remove_baseline_median(ecg_signal, fs)
    % REMOVE_BASELINE_MEDIAN Two-stage median filter baseline wander removal.
    %
    % Method:
    %   1. Stage 1: Median filter with 200 ms window (removes QRS & P/T waves).
    %   2. Stage 2: Median filter with 600 ms window (extracts low-frequency baseline).
    %   3. Subtract estimated baseline from filtered ECG.
    %
    % Inputs:
    %   ecg_signal - ECG signal vector (pre-filtered by bandpass filter)
    %   fs         - Sampling rate in Hz (default: 100 Hz)
    %
    % Output:
    %   clean_ecg  - Baseline-subtracted ECG signal

    if nargin < 2
        fs = 100;
    end

    % Window sizes in samples
    w1 = round(0.200 * fs); % 200 ms window
    w2 = round(0.600 * fs); % 600 ms window

    % Ensure odd window lengths for median filtering
    if mod(w1, 2) == 0, w1 = w1 + 1; end
    if mod(w2, 2) == 0, w2 = w2 + 1; end

    % Stage 1: 200 ms median filter
    temp = medfilt1(ecg_signal, w1);

    % Stage 2: 600 ms median filter to isolate baseline wander
    baseline = medfilt1(temp, w2);

    % Subtract baseline drift
    clean_ecg = ecg_signal - baseline;
end
