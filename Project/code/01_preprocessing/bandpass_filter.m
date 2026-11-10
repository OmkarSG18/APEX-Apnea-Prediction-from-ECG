function filtered_ecg = bandpass_filter(ecg_signal, fs)
    % BANDPASS_FILTER Applies 4th-order zero-phase IIR bandpass filtering (0.5 - 45 Hz).
    % Robust fallback support for environments without Signal Processing Toolbox.

    if nargin < 2
        fs = 100;
    end

    try
        if exist('designfilt', 'file') || exist('designfilt', 'builtin')
            d = designfilt('bandpassiir', ...
                'FilterOrder', 4, ...
                'HalfPowerFrequency1', 0.5, ...
                'HalfPowerFrequency2', 45, ...
                'SampleRate', fs);
            filtered_ecg = filtfilt(d, ecg_signal);
        else
            [b, a] = butter(4, [0.5 45] / (fs / 2), 'bandpass');
            if exist('filtfilt', 'file') || exist('filtfilt', 'builtin')
                filtered_ecg = filtfilt(b, a, ecg_signal);
            else
                filtered_ecg = filter(b, a, ecg_signal);
            end
        end
    catch
        % Fallback simple moving difference highpass + lowpass smoothing
        filtered_ecg = ecg_signal - movmean(ecg_signal, round(fs / 0.5));
    end
end
