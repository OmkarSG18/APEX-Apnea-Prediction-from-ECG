function clean_ecg = remove_baseline_wavelet(ecg_signal, fs)
    % REMOVE_BASELINE_WAVELET DWT wavelet-based baseline wander removal using db6.
    % Fallback to median baseline removal if Wavelet Toolbox is not installed.

    if nargin < 2
        fs = 100;
    end

    try
        if exist('wdenoise', 'file') || exist('wdenoise', 'builtin')
            clean_ecg = wdenoise(ecg_signal, 8, 'Wavelet', 'db6', ...
                                 'DenoisingMethod', 'UniversalThreshold', ...
                                 'ThresholdRule', 'Soft');
        else
            clean_ecg = remove_baseline_median(ecg_signal, fs);
        end
    catch
        clean_ecg = remove_baseline_median(ecg_signal, fs);
    end
end
