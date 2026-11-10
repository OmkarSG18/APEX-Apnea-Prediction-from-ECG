function validate_rpeaks()
    % VALIDATE_RPEAKS Validates Pan-Tompkins R-peak detection accuracy against reference .qrs annotations.
    % Generates results/tables/rpeak_validation.csv.

    data_dir = fullfile('..', '..', 'data', 'apnea-ecg');
    results_dir = fullfile('..', '..', 'results', 'tables');
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end

    csv_file = fullfile(results_dir, 'rpeak_validation.csv');
    fid = fopen(csv_file, 'w');
    fprintf(fid, 'Record,Ref_Beats,Detected_Beats,Matched_Beats,Sensitivity,Precision,F1_Score\n');

    % Records with reference .qrs annotations available
    qrs_records = {'a01', 'a02', 'a03', 'a04', 'b01', 'c01'};

    fprintf('Validating Pan-Tompkins R-peak detector against reference annotations...\n');

    for i = 1:length(qrs_records)
        rec = qrs_records{i};
        rec_path = fullfile(data_dir, rec);
        qrs_file = fullfile(data_dir, [rec, '.qrs']);

        if exist([rec_path, '.dat'], 'file') && exist(qrs_file, 'file')
            try
                % Read raw ECG signal and reference QRS annotations via WFDB
                [signal, fs] = rdsamp(rec_path, 1);
                [ref_locs, ~] = rdann(rec_path, 'qrs');

                % Preprocess ECG signal
                filtered = bandpass_filter(signal, fs);
                clean = remove_baseline_median(filtered, fs);

                % Detect R-peaks using Pan-Tompkins
                [~, det_locs, ~] = pan_tompkins(clean, fs);

                % Compute matching metrics within ±50 ms (5 samples @ 100 Hz) tolerance window
                tol = round(0.050 * fs);
                n_ref = length(ref_locs);
                n_det = length(det_locs);
                matched = 0;

                for j = 1:n_ref
                    if any(abs(det_locs - ref_locs(j)) <= tol)
                        matched = matched + 1;
                    end
                end

                sens = matched / n_ref;
                prec = matched / n_det;
                f1 = 2 * (sens * prec) / (sens + prec);

                fprintf('  Record %s: Sens=%.2f%%, Prec=%.2f%%, F1=%.2f%%\n', rec, sens*100, prec*100, f1*100);
                fprintf(fid, '%s,%d,%d,%d,%.4f,%.4f,%.4f\n', rec, n_ref, n_det, matched, sens, prec, f1);
            catch ME
                fprintf('  [SKIP] Record %s validation failed: %s\n', rec, ME.message);
            end
        end
    end

    fclose(fid);
    fprintf('R-peak validation completed. Saved to: %s\n', csv_file);
end
