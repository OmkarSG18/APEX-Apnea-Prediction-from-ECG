function check_dataset_files()
    % CHECK_DATASET_FILES Verification script for PhysioNet Apnea-ECG dataset completeness.
    % Scans data/apnea-ecg/ and verifies .dat, .hea, and .apn presence for all 70 records.

    data_dir = fullfile('..', '..', 'data', 'apnea-ecg');
    if ~exist(data_dir, 'dir')
        error('Data directory does not exist: %s. Run download_apnea_ecg first.', data_dir);
    end

    train_records = [ ...
        arrayfun(@(x) sprintf('a%02d', x), 1:20, 'UniformOutput', false), ...
        arrayfun(@(x) sprintf('b%02d', x), 1:5, 'UniformOutput', false), ...
        arrayfun(@(x) sprintf('c%02d', x), 1:10, 'UniformOutput', false) ...
    ];
    test_records = arrayfun(@(x) sprintf('x%02d', x), 1:35, 'UniformOutput', false);
    all_records = [train_records, test_records];

    missing_count = 0;
    fprintf('Verifying dataset files in %s...\n', data_dir);

    for i = 1:length(all_records)
        rec = all_records{i};
        dat_ok = exist(fullfile(data_dir, [rec, '.dat']), 'file');
        hea_ok = exist(fullfile(data_dir, [rec, '.hea']), 'file');
        apn_ok = exist(fullfile(data_dir, [rec, '.apn']), 'file');

        if ~(dat_ok && hea_ok && apn_ok)
            fprintf('  [MISSING] Record %s (DAT:%d, HEA:%d, APN:%d)\n', rec, dat_ok, hea_ok, apn_ok);
            missing_count = missing_count + 1;
        end
    end

    if missing_count == 0
        fprintf('  [SUCCESS] All %d records contain valid .dat, .hea, and .apn files.\n', length(all_records));
    else
        fprintf('  [WARNING] %d records are missing required files.\n', missing_count);
    end
end
