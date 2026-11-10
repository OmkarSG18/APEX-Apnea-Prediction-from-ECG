function build_feature_matrix()
    % BUILD_FEATURE_MATRIX Loads or builds feature matrices for train/test splits.
    % First checks if pre-computed .mat files exist and are valid.
    % If valid, loads them directly (no dataset/WFDB needed).
    % Otherwise, attempts to build from raw ECG files.

    % Search multiple candidate paths and VALIDATE each file has real data
    results_candidates = {
        fullfile('..', 'results'), ...          % one level up:  Project/results (Python output)
        fullfile('..', '..', 'results'), ...   % two levels up: OE-2/results
        fullfile(fileparts(fileparts(mfilename('fullpath'))), '..', 'results') % relative to this file
    };

    results_dir = fullfile('..', '..', 'results'); % default for saving new files
    precomp_train = '';
    precomp_test  = '';

    for ci = 1:length(results_candidates)
        cand_train = fullfile(results_candidates{ci}, 'feature_matrix_train.mat');
        cand_test  = fullfile(results_candidates{ci}, 'feature_matrix_test.mat');
        if exist(cand_train, 'file') && exist(cand_test, 'file')
            try
                tmp = load(cand_train);
                if isfield(tmp, 'X_train') && size(tmp.X_train, 1) > 10 && size(tmp.X_train, 2) > 1
                    precomp_train = cand_train;
                    precomp_test  = cand_test;
                    results_dir   = results_candidates{ci};
                    fprintf('  [INFO] Found valid pre-computed matrices in: %s\n', results_candidates{ci});
                    break;
                else
                    fprintf('  [SKIP] File exists but has no valid data: %s\n', cand_train);
                end
            catch
                fprintf('  [SKIP] Could not read: %s\n', cand_train);
            end
        end
    end

    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end

    train_file = fullfile(results_dir, 'feature_matrix_train.mat');
    test_file  = fullfile(results_dir, 'feature_matrix_test.mat');

    % -----------------------------------------------------------------------
    % Priority 1: Load pre-computed matrices if they exist and are populated
    % -----------------------------------------------------------------------
    if ~isempty(precomp_train)
        try
            tr = load(precomp_train);
            te = load(precomp_test);

            if isfield(tr, 'X_train') && isfield(te, 'X_test') && ...
               numel(tr.X_train) > 0 && numel(te.X_test) > 0 && ...
               size(tr.X_train, 2) > 1 && size(te.X_test, 2) > 1

                fprintf('  [OK] Pre-computed feature matrices loaded from disk:\n');
                fprintf('       Train: %d segments x %d features\n', size(tr.X_train,1), size(tr.X_train,2));
                fprintf('       Test:  %d segments x %d features\n', size(te.X_test,1), size(te.X_test,2));
                return;
            end
        catch ME
            fprintf('  [WARN] Could not load pre-computed files: %s\n', ME.message);
        end
    end

    % -----------------------------------------------------------------------
    % Priority 2: Build feature matrices from raw ECG files
    % -----------------------------------------------------------------------
    data_dir = fullfile('..', '..', 'data', 'apnea-ecg');

    train_records = [ ...
        arrayfun(@(x) sprintf('a%02d', x), 1:20, 'UniformOutput', false), ...
        arrayfun(@(x) sprintf('b%02d', x), 1:5, 'UniformOutput', false), ...
        arrayfun(@(x) sprintf('c%02d', x), 1:10, 'UniformOutput', false) ...
    ];
    test_records = arrayfun(@(x) sprintf('x%02d', x), 1:35, 'UniformOutput', false);

    fprintf('Processing Training Split (%d records)...\n', length(train_records));
    [X_train_raw, y_train] = extract_split_features(train_records, data_dir);

    fprintf('Processing Test Split (%d records)...\n', length(test_records));
    [X_test_raw, y_test] = extract_split_features(test_records, data_dir);

    if isempty(X_train_raw)
        error('No dataset records could be read. Please run:\n  python code/utils/fast_download.py\nto download the dataset, then run main_pipeline again.');
    end

    % Impute NaNs
    med_vals = median(X_train_raw, 1, 'omitnan');
    med_vals(isnan(med_vals)) = 0;
    for col = 1:size(X_train_raw, 2)
        X_train_raw(isnan(X_train_raw(:, col)), col) = med_vals(col);
        if ~isempty(X_test_raw)
            X_test_raw(isnan(X_test_raw(:, col)), col) = med_vals(col);
        end
    end

    % Z-score standardize using training stats only
    mu    = mean(X_train_raw, 1);
    sigma = std(X_train_raw, 0, 1);
    sigma(sigma == 0) = 1;

    X_train = (X_train_raw - mu) ./ sigma;
    if ~isempty(X_test_raw)
        X_test = (X_test_raw - mu) ./ sigma;
    else
        split_idx = round(0.8 * size(X_train, 1));
        X_test = X_train(split_idx+1:end, :);
        y_test = y_train(split_idx+1:end);
        X_train = X_train(1:split_idx, :);
        y_train = y_train(1:split_idx);
    end

    feature_names = { ...
        'mRR','SDNN','RMSSD','pNN50','HR_mean', ...
        'LF_power','HF_power','LF_HF_ratio', ...
        'Wavelet_Entropy','Energy_A5','Ratio_D3_D4','Var_D4', ...
        'EDR_Amplitude','EDR_Rate' ...
    };

    save(train_file, 'X_train', 'y_train', 'feature_names');
    save(test_file, 'X_test', 'y_test', 'feature_names');

    fprintf('Feature matrices built and saved:\n');
    fprintf('  Train: %d segments x %d features -> %s\n', size(X_train,1), size(X_train,2), train_file);
    fprintf('  Test:  %d segments x %d features -> %s\n', size(X_test,1), size(X_test,2), test_file);
end


function [X_matrix, y_vector] = extract_split_features(records, data_dir)
    X_matrix = [];
    y_vector  = {};
    fs = 100;
    segment_len = 60 * fs;

    for r = 1:length(records)
        rec = records{r};
        dat_file = fullfile(data_dir, [rec, '.dat']);
        apn_file = fullfile(data_dir, [rec, '.apn']);

        if ~exist(dat_file, 'file') || ~exist(apn_file, 'file')
            continue;
        end

        try
            if ~isempty(which('rdsamp'))
                [signal, ~] = rdsamp(fullfile(data_dir, rec), 1);
            else
                fid = fopen(dat_file, 'r', 'ieee-le');
                signal = double(fread(fid, Inf, 'int16'));
                fclose(fid);
            end

            if ~isempty(which('rdann'))
                [~, ann_type] = rdann(fullfile(data_dir, rec), 'apn');
                ann_vec = cellstr(ann_type);
            else
                fid = fopen(apn_file, 'r');
                raw = fread(fid, Inf, 'uint8=>char')';
                fclose(fid);
                valid = (raw == 'A') | (raw == 'N');
                raw = raw(valid);
                ann_vec = cellstr(raw(:));
            end

            filtered = bandpass_filter(signal, fs);
            clean    = remove_baseline_median(filtered, fs);
            [r_peaks, r_locs, ~] = pan_tompkins(clean, fs);

            n_minutes = min(floor(length(clean) / segment_len), length(ann_vec));

            for m = 1:n_minutes
                seg_start = (m-1)*segment_len + 1;
                seg_end   = m*segment_len;
                ecg_seg   = clean(seg_start:seg_end);

                mask   = (r_locs >= seg_start) & (r_locs <= seg_end);
                m_peaks = r_peaks(mask);
                m_locs  = r_locs(mask) - seg_start + 1;
                m_rr    = diff(m_locs) / fs;
                m_rr_t  = m_locs(2:end) / fs;

                row = [extract_time_hrv(m_rr), ...
                       extract_freq_hrv(m_rr, m_rr_t, 4), ...
                       extract_wavelet(ecg_seg, fs), ...
                       extract_edr(m_peaks, m_locs, fs)];

                X_matrix = [X_matrix; row]; %#ok<AGROW>
                y_vector = [y_vector; ann_vec(m)]; %#ok<AGROW>
            end
        catch
            % Skip unreadable record silently
        end
    end
end
