function check_environment()
    % CHECK_ENVIRONMENT Verification script for MATLAB version & required toolboxes.
    % Outputs diagnostic logs to results/setup_verification.txt.

    fprintf('====================================================\n');
    fprintf('  Sleep Apnea ECG Project - MATLAB Environment Check\n');
    fprintf('====================================================\n\n');

    % Log location
    results_dir = fullfile('..', '..', 'results');
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end
    log_file = fullfile(results_dir, 'setup_verification.txt');
    fid = fopen(log_file, 'w');

    % MATLAB version
    matlab_ver = version;
    fprintf('MATLAB Version: %s\n', matlab_ver);
    fprintf(fid, 'MATLAB Version: %s\n\n', matlab_ver);

    % Required Toolboxes
    required_toolboxes = { ...
        'Signal Processing Toolbox', ...
        'Wavelet Toolbox', ...
        'Statistics and Machine Learning Toolbox' ...
    };

    installed_ver = ver;
    installed_names = {installed_ver.Name};

    all_present = true;
    fprintf('Checking Required Toolboxes:\n');
    fprintf(fid, 'Checking Required Toolboxes:\n');

    for i = 1:length(required_toolboxes)
        tb_name = required_toolboxes{i};
        tf = ismember(tb_name, installed_names);
        if tf
            fprintf('  [OK]   %s\n', tb_name);
            fprintf(fid, '  [OK]   %s\n', tb_name);
        else
            fprintf('  [MISSING] %s\n', tb_name);
            fprintf(fid, '  [MISSING] %s\n', tb_name);
            all_present = false;
        end
    end

    % Check WFDB Toolbox installation on Path
    wfdb_installed = ~isempty(which('rdsamp')) && ~isempty(which('rdann'));
    if wfdb_installed
        fprintf('  [OK]   WFDB Toolbox for MATLAB\n');
        fprintf(fid, '  [OK]   WFDB Toolbox for MATLAB\n');
    else
        fprintf('  [WARN] WFDB Toolbox for MATLAB not detected on path.\n');
        fprintf('         (Install from PhysioNet or add to MATLAB path using addpath).\n');
        fprintf(fid, '  [WARN] WFDB Toolbox for MATLAB not detected on path.\n');
    end

    fprintf('\nEnvironment check completed. Log written to: %s\n', log_file);
    fclose(fid);
end
