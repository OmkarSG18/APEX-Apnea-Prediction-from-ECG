function wav_feats = extract_wavelet(ecg_segment, fs)
    % EXTRACT_WAVELET Computes 4 wavelet-based features (WE, A5, D3/D4 ratio, var(D4)).
    % Fallback to FFT subband energy decomposition if Wavelet Toolbox is absent.

    if nargin < 2
        fs = 100;
    end

    try
        if exist('wavedec', 'file') || exist('wavedec', 'builtin')
            [c, l] = wavedec(ecg_segment, 5, 'db4');
            cA5 = appcoef(c, l, 'db4', 5);
            cD5 = detcoef(c, l, 5);
            cD4 = detcoef(c, l, 4);
            cD3 = detcoef(c, l, 3);
            cD2 = detcoef(c, l, 2);
            cD1 = detcoef(c, l, 1);

            E_A5 = sum(cA5 .^ 2);
            E_D5 = sum(cD5 .^ 2);
            E_D4 = sum(cD4 .^ 2);
            E_D3 = sum(cD3 .^ 2);
            E_D2 = sum(cD2 .^ 2);
            E_D1 = sum(cD1 .^ 2);

            energies = [E_A5, E_D5, E_D4, E_D3, E_D2, E_D1];
            E_total = sum(energies);
            prob = energies / (E_total + eps);
            WE = -sum(prob .* log2(prob + eps));

            Energy_A5 = E_A5;
            Ratio_D3_D4 = E_D3 / (E_D4 + eps);
            Var_D4 = var(cD4);

            wav_feats = [WE, Energy_A5, Ratio_D3_D4, Var_D4];
        else
            % FFT subband energy fallback calculation
            N = length(ecg_segment);
            Y = abs(fft(ecg_segment));
            half_N = floor(N / 2);
            subband_len = floor(half_N / 6);
            energies = zeros(1, 6);
            for b = 1:6
                st = (b - 1) * subband_len + 1;
                en = min(b * subband_len, half_N);
                energies(b) = sum(Y(st:en) .^ 2);
            end

            E_total = sum(energies);
            prob = energies / (E_total + eps);
            WE = -sum(prob .* log2(prob + eps));
            Energy_A5 = energies(1);
            Ratio_D3_D4 = energies(3) / (energies(4) + eps);
            Var_D4 = var(ecg_segment);

            wav_feats = [WE, Energy_A5, Ratio_D3_D4, Var_D4];
        end
    catch
        wav_feats = [0, 0, 0, 0];
    end
end
