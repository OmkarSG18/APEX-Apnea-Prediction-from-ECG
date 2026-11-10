function freq_feats = extract_freq_hrv(rr_clean, rr_times, fs_resample)
    % EXTRACT_FREQ_HRV Computes 3 frequency-domain HRV features (LF, HF, LF/HF).
    % Fallback to FFT-based power spectrum if Signal Processing Toolbox (pwelch) is absent.

    if nargin < 3
        fs_resample = 4;
    end

    valid_idx = ~isnan(rr_clean);
    rr = rr_clean(valid_idx);
    times = rr_times(valid_idx);

    if length(rr) < 4 || (times(end) - times(1)) < 10
        freq_feats = [NaN, NaN, NaN];
        return;
    end

    try
        t_uniform = times(1):(1/fs_resample):times(end);
        rr_uniform = interp1(times, rr, t_uniform, 'spline');
        rr_detrend = rr_uniform - mean(rr_uniform);

        N = length(rr_detrend);

        if exist('pwelch', 'file') || exist('pwelch', 'builtin')
            win_len = min(N, round(30 * fs_resample));
            noverlap = round(win_len / 2);
            nfft = max(256, 2^nextpow2(win_len));
            [psd, f] = pwelch(rr_detrend, win_len, noverlap, nfft, fs_resample);
        else
            % Native FFT fallback calculation for PSD
            nfft = 2^nextpow2(N);
            Y = fft(rr_detrend, nfft);
            P2 = abs(Y / N);
            psd = P2(1:floor(nfft/2)+1);
            psd(2:end-1) = 2 * psd(2:end-1);
            f = fs_resample * (0:(nfft/2)) / nfft;
        end

        lf_mask = (f >= 0.04) & (f <= 0.15);
        hf_mask = (f >= 0.15) & (f <= 0.40);

        LF_power = trapz(f(lf_mask), psd(lf_mask));
        HF_power = trapz(f(hf_mask), psd(hf_mask));

        if HF_power > 0
            LF_HF_ratio = LF_power / HF_power;
        else
            LF_HF_ratio = NaN;
        end

        freq_feats = [LF_power, HF_power, LF_HF_ratio];
    catch
        freq_feats = [NaN, NaN, NaN];
    end
end
