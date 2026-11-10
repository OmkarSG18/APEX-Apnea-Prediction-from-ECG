function edr_feats = extract_edr(r_peaks, r_locs, fs)
    % EXTRACT_EDR Computes 2 ECG-Derived Respiration (EDR) features per minute.
    %
    % Method:
    %   1. Interpolates R-peak amplitudes to a uniform 4 Hz grid.
    %   2. Computes EDR Peak-to-Peak Amplitude modulation.
    %   3. Estimates respiratory rate (breaths per minute).
    %
    % Inputs:
    %   r_peaks   - Vector of R-peak amplitudes
    %   r_locs    - Sample indices of R-peaks
    %   fs        - Sampling rate (default: 100 Hz)
    % Output:
    %   edr_feats - 1x2 vector [EDR_Amplitude, EDR_Rate]

    if nargin < 3
        fs = 100;
    end

    if length(r_peaks) < 4
        edr_feats = [NaN, NaN];
        return;
    end

    try
        r_times = r_locs / fs;
        t_grid = r_times(1):(1/4):r_times(end);
        edr_sig = interp1(r_times, r_peaks, t_grid, 'spline');

        % Feature 1: EDR Peak-to-Peak amplitude modulation
        EDR_Amplitude = max(edr_sig) - min(edr_sig);

        % Feature 2: EDR breathing rate (peaks per minute)
        [pks, ~] = findpeaks(edr_sig, 'MinPeakDistance', 4 * 1.5); % > 1.5 s between breaths
        EDR_Rate = length(pks);

        edr_feats = [EDR_Amplitude, EDR_Rate];
    catch
        edr_feats = [NaN, NaN];
    end
end
