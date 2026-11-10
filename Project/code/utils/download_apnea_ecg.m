function download_apnea_ecg()
    % DOWNLOAD_APNEA_ECG Automated downloader for PhysioNet Apnea-ECG Database records.
    % Downloads dataset files (.dat, .hea, .apn, .qrs) into data/apnea-ecg/

    target_dir = fullfile('..', '..', 'data', 'apnea-ecg');
    if ~exist(target_dir, 'dir')
        mkdir(target_dir);
    end

    base_url = 'https://physionet.org/files/apnea-ecg/1.0.0/';
    
    % Record names (35 training records: a01-a20, b01-b05, c01-c10; 35 test records: x01-x35)
    train_records = [ ...
        arrayfun(@(x) sprintf('a%02d', x), 1:20, 'UniformOutput', false), ...
        arrayfun(@(x) sprintf('b%02d', x), 1:5, 'UniformOutput', false), ...
        arrayfun(@(x) sprintf('c%02d', x), 1:10, 'UniformOutput', false) ...
    ];
    test_records = arrayfun(@(x) sprintf('x%02d', x), 1:35, 'UniformOutput', false);
    all_records = [train_records, test_records];

    extensions = {'.dat', '.hea', '.apn'};

    fprintf('Downloading PhysioNet Apnea-ECG Database (%d records)...', length(all_records));
    
    for i = 1:length(all_records)
        rec = all_records{i};
        for j = 1:length(extensions)
            ext = extensions{j};
            filename = [rec, ext];
            local_path = fullfile(target_dir, filename);
            if ~exist(local_path, 'file')
                remote_url = [base_url, filename];
                try
                    websave(local_path, remote_url);
                catch
                    % Fallback to urlwrite if websave encounters issue
                    try
                        urlwrite(remote_url, local_path);
                    catch ME
                        fprintf('\n[ERROR] Failed to download %s: %s', filename, ME.message);
                    end
                end
            end
        end
        % Optional .qrs annotations if available
        qrs_filename = [rec, '.qrs'];
        qrs_path = fullfile(target_dir, qrs_filename);
        if ~exist(qrs_path, 'file')
            try
                websave(qrs_path, [base_url, qrs_filename]);
            catch
                % qrs file may not exist for all records
            end
        end
    end
    fprintf('\nDownload process finished. Data saved in: %s\n', target_dir);
end
