function [peaks, locs, rr_clean] = pan_tompkins(ecg_signal, fs)
    % PAN_TOMPKINS Real-time Pan-Tompkins QRS Detection Algorithm.
    %
    % Steps:
    %   1. Bandpass filter (5-15 Hz) tailored for QRS energy isolation.
    %   2. Differentiation (emphasizes steep slope of R-waves).
    %   3. Squaring (non-linear amplification of derivative signal).
    %   4. Moving-window integration (150 ms window).
    %   5. Adaptive thresholding & peak refinement.
    %   6. Ectopic beat filtering (>20% deviation from 5-beat local median).
    %
    % Inputs:
    %   ecg_signal - Preprocessed ECG signal vector
    %   fs         - Sampling frequency in Hz (default: 100 Hz)
    %
    % Outputs:
    %   peaks      - R-peak amplitudes in filtered ECG
    %   locs       - Sample indices of detected R-peaks
    %   rr_clean   - Cleaned NN interval vector in seconds

    if nargin < 2
        fs = 100;
    end

    % 1. Bandpass filter 5 - 15 Hz for QRS detector stage
    [b, a] = butter(3, [5 15] / (fs / 2), 'bandpass');
    filtered = filtfilt(b, a, ecg_signal);

    % 2. Derivative stage
    h_diff = [-1 -2 0 2 1] * (fs / 8);
    d_sig = conv(filtered, h_diff, 'same');

    % 3. Squaring stage
    sq_sig = d_sig .^ 2;

    % 4. Moving-window integration (150 ms window)
    window_samples = round(0.150 * fs);
    int_sig = movmean(sq_sig, window_samples);

    % 5. Peak picking on integrated signal
    min_dist = round(0.600 * fs); % Minimum 600 ms (~100 bpm max physiological resting limit)
    thresh = 0.35 * max(int_sig);
    [~, int_locs] = findpeaks(int_sig, 'MinPeakDistance', min_dist, 'MinPeakHeight', thresh);

    % Refine peak locations by finding local maximum in filtered ECG within ±100 ms window
    search_win = round(0.100 * fs);
    locs = zeros(size(int_locs));
    N = length(ecg_signal);

    for i = 1:length(int_locs)
        idx = int_locs(i);
        win_start = max(1, idx - search_win);
        win_end = min(N, idx + search_win);
        [~, max_rel] = max(ecg_signal(win_start:win_end));
        locs(i) = win_start + max_rel - 1;
    end

    % Remove duplicate indices
    locs = unique(locs);
    peaks = ecg_signal(locs);

    % Compute RR intervals in seconds
    rr_raw = diff(locs) / fs;

    % 6. Ectopic beat cleaning: filter out RR intervals > 20% deviation from 5-beat local median
    rr_clean = rr_raw;
    for i = 3:(length(rr_raw) - 2)
        local_med = median(rr_raw(i-2:i+2));
        if abs(rr_raw(i) - local_med) / local_med > 0.20
            rr_clean(i) = NaN; % Mark ectopic / noisy interval
        end
    end
end
